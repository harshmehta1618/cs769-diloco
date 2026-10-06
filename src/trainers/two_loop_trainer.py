"""
Unified Two-Loop Trainer for DiLoCo / SNOO / PeriodicAvg.

This single trainer implements all DiLoCo-family methods by switching on
configuration flags:

  method="snoo"        → M=1 worker, outer optimizer = Nesterov
  method="periodic_avg"→ M workers, outer optimizer = plain averaging
  method="diloco"      → M workers, outer optimizer = Nesterov
  method="clone_diloco"→ M workers (clone shard mode), outer = Nesterov

Algorithm (per outer step t):
  1.  Save reference parameters θ_ref  (all workers share this)
  2.  For each worker w = 1..M  (independent inner loops):
        For inner step h = 1..H:
          sample batch from worker_w's shard
          compute loss, backward
          inner_optimizer.step()       # AdamW
  3.  Compute pseudo-gradient for each worker:
        Δ_w = θ_ref − θ_w
  4.  Average:  avg_Δ = mean_w(Δ_w)
  5.  Apply outer optimizer:
        θ_new = outer_opt.step(θ_ref, avg_Δ)
  6.  Copy θ_new back to all worker models
  7.  (Optionally) reset inner-optimizer state

Communication is logged for each outer step via CommunicationTracker.
"""

from __future__ import annotations

import copy
from typing import Dict, Iterator, List, Optional

import torch
import torch.nn as nn

from .base_trainer import BaseTrainer
from ..metrics.comm_tracker import CommunicationTracker
from ..metrics.geometry import compute_all_geometry_metrics
from ..optimizers.inner import build_inner_optimizer, update_lr
from ..optimizers.outer import build_outer_optimizer
from ..utils.config import RunConfig


class TwoLoopTrainer(BaseTrainer):
    """
    Unified trainer for SNOO, PeriodicAvg, DiLoCo, and Clone-DiLoCo.

    Parameters
    ----------
    cfg          : RunConfig
    model        : reference model (will be deep-copied for each worker)
    train_iters  : {worker_idx: data_iterator}
    eval_iter    : validation data iterator
    device       : torch.device
    log_geometry : if True, compute and log worker-diversity geometry metrics
                   at every outer step (slower; useful for E3)
    """

    def __init__(
        self,
        cfg: RunConfig,
        model: nn.Module,
        train_iters: Dict[int, Iterator],
        eval_iter: Iterator,
        device: torch.device,
        log_geometry: bool = False,
    ) -> None:
        super().__init__(cfg, model, train_iters, eval_iter, device)
        self.log_geometry = log_geometry

        M   = cfg.distribution.M_workers
        io  = cfg.inner_optimizer
        oo  = cfg.outer_optimizer

        # ----------------------------------------------------------------
        # Worker models — each worker has its own model copy
        # ----------------------------------------------------------------
        self.worker_models: List[nn.Module] = [
            copy.deepcopy(model).to(device) for _ in range(M)
        ]

        # ----------------------------------------------------------------
        # Worker inner optimizers — one per worker
        # ----------------------------------------------------------------
        self.worker_optimizers: List[torch.optim.Optimizer] = [
            build_inner_optimizer(
                model=self.worker_models[w],
                lr=io.lr,
                betas=io.betas,
                eps=io.eps,
                weight_decay=io.weight_decay,
            )
            for w in range(M)
        ]

        # ----------------------------------------------------------------
        # Outer optimizer
        # ----------------------------------------------------------------
        outer_type = oo.type   # "nesterov" | "avg" | "none"
        self.outer_optimizer = build_outer_optimizer(
            outer_type=outer_type,
            ref_params_iter=iter(model.named_parameters()) if outer_type == "nesterov" else None,
            outer_lr=oo.outer_lr,
            momentum=oo.momentum,
            nesterov=oo.nesterov,
        )

        # Keep a separate comm tracker (base class also has one, but this is clearer)
        self.comm_tracker = CommunicationTracker()

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def _save_ref_params(self) -> Dict[str, torch.Tensor]:
        """Snapshot reference parameter values (θ_ref) before inner loops."""
        return {
            name: p.data.clone()
            for name, p in self.model.named_parameters()
        }

    def _sync_workers_from_ref(self, ref_params: Dict[str, torch.Tensor]) -> None:
        """Copy θ_ref into all worker models at the start of each outer step."""
        for worker_model in self.worker_models:
            for name, p in worker_model.named_parameters():
                p.data.copy_(ref_params[name])

    def _run_inner_loop(self, worker_idx: int) -> None:
        """
        Run H inner AdamW steps on worker worker_idx.
        Advances token counter and LR schedule.
        """
        H    = self.cfg.distribution.H_inner_steps
        io   = self.cfg.inner_optimizer
        wm   = self.worker_models[worker_idx]
        wopt = self.worker_optimizers[worker_idx]
        wm.train()
        wopt.zero_grad(set_to_none=True)

        for h in range(H):
            input_ids, targets = self._next_batch(worker_idx)
            _, loss = wm(input_ids, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(wm.parameters(), io.grad_clip)
            # Token-driven LR update (same formula as DP-AdamW)
            n_tok = self._count_tokens(targets)
            self.token_counter.step(n_tok)
            update_lr(
                optimizer=wopt,
                total_tokens=self.token_counter.total_tokens,
                token_budget=self.token_counter.token_budget,
                peak_lr=io.lr,
                warmup_tokens=io.warmup_tokens,
                min_lr_ratio=io.min_lr_ratio,
            )
            wopt.step()
            wopt.zero_grad(set_to_none=True)

    def _compute_pseudo_gradients(
        self, ref_params: Dict[str, torch.Tensor]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Δ_w = θ_ref − θ_w   for each worker w.
        """
        pseudo_grads = []
        for wm in self.worker_models:
            pg = {
                name: ref_params[name] - p.data
                for name, p in wm.named_parameters()
            }
            pseudo_grads.append(pg)
        return pseudo_grads

    def _average_pseudo_gradients(
        self, pseudo_grads: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """avg_Δ = mean_w(Δ_w)"""
        M = len(pseudo_grads)
        avg = {}
        for name in pseudo_grads[0]:
            avg[name] = sum(pg[name] for pg in pseudo_grads) / M
        return avg

    def _apply_outer_step(
        self,
        ref_params:      Dict[str, torch.Tensor],
        avg_pseudo_grad: Dict[str, torch.Tensor],
        outer_step:      int,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply outer optimizer to get new parameters, log comm bytes.
        """
        with self.comm_tracker.record(outer_step, self.token_counter.total_tokens) as rec:
            rec.set_payload(self.model)   # bytes = param count * element size

            if self.outer_optimizer is not None:
                new_params = self.outer_optimizer.step(ref_params, avg_pseudo_grad)
            else:
                # No outer optimizer: revert to ref (shouldn't happen in practice)
                new_params = ref_params

        return new_params

    def _broadcast_params(self, new_params: Dict[str, torch.Tensor]) -> None:
        """Copy new parameters into reference model and all worker models."""
        # Update reference model
        for name, p in self.model.named_parameters():
            p.data.copy_(new_params[name])
        # Broadcast to workers
        for wm in self.worker_models:
            for name, p in wm.named_parameters():
                p.data.copy_(new_params[name])

    def _maybe_reset_inner_optimizers(self) -> None:
        """Optionally reset inner-optimizer state after each outer step."""
        if self.cfg.outer_optimizer.reset_inner:
            io = self.cfg.inner_optimizer
            for w, wm in enumerate(self.worker_models):
                self.worker_optimizers[w] = build_inner_optimizer(
                    model=wm, lr=io.lr, betas=io.betas,
                    eps=io.eps, weight_decay=io.weight_decay,
                )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> dict:
        """Run the full outer-loop training until token budget is exhausted."""
        outer_step = 0

        while not self.token_counter.budget_exhausted:
            outer_step += 1
            self._global_step = outer_step

            # Step 1: save reference parameters
            ref_params = self._save_ref_params()

            # Step 2: sync all workers from ref
            self._sync_workers_from_ref(ref_params)

            # Step 3: run H inner steps on each worker
            for w in range(self.cfg.distribution.M_workers):
                if self.token_counter.budget_exhausted:
                    break
                self._run_inner_loop(w)

            if self.token_counter.budget_exhausted and outer_step == 1:
                # Budget too small for even one outer step — still update
                pass

            # Step 4: compute pseudo-gradients
            pseudo_grads    = self._compute_pseudo_gradients(ref_params)
            avg_pseudo_grad = self._average_pseudo_gradients(pseudo_grads)

            # Step 5: apply outer optimizer
            new_params = self._apply_outer_step(ref_params, avg_pseudo_grad, outer_step)

            # Step 6: broadcast
            self._broadcast_params(new_params)

            # Step 7: reset inner optimizers if configured
            self._maybe_reset_inner_optimizers()

            # Compute train loss (from last worker's last step proxy)
            # We log the average of a fresh eval-step on the updated model
            with torch.no_grad():
                self.model.eval()
                inp, tgt = self._next_batch(0)
                _, train_loss = self.model(inp, tgt)
                train_loss = train_loss.item()
                self.model.train()

            # Geometry metrics (E3)
            geo_metrics = {}
            if self.log_geometry:
                worker_params = [
                    {n: p.data for n, p in wm.named_parameters()}
                    for wm in self.worker_models
                ]
                velocity = (
                    self.outer_optimizer._velocity
                    if hasattr(self.outer_optimizer, "_velocity") else None
                )
                geo_metrics = compute_all_geometry_metrics(
                    pseudo_grads=pseudo_grads,
                    worker_params=worker_params,
                    avg_pseudo_grad=avg_pseudo_grad,
                    velocity=velocity,
                )

            self.logger.log(
                step=outer_step,
                total_tokens=self.token_counter.total_tokens,
                train_loss=train_loss,
                comm_bytes_total=self.comm_tracker.total_bytes,
                **geo_metrics,
            )

            # Evaluation
            eval_metrics = self._maybe_evaluate()
            if eval_metrics:
                self.logger.log(
                    step=outer_step,
                    total_tokens=self.token_counter.total_tokens,
                    **eval_metrics,
                )

            # Checkpoint
            outer_opt_state = (
                self.outer_optimizer.state_dict()
                if self.outer_optimizer is not None else None
            )
            self._maybe_checkpoint(outer_opt_state)

        # Final evaluation (primary result = fixed final token budget)
        final_metrics = self.evaluator.evaluate(self.model)
        self.logger.log(
            step=outer_step,
            total_tokens=self.token_counter.total_tokens,
            **{f"final_{k}": v for k, v in final_metrics.items()},
        )
        self.logger.finish()

        return {
            "method":           self.cfg.distribution.method,
            "M_workers":        self.cfg.distribution.M_workers,
            "H_inner_steps":    self.cfg.distribution.H_inner_steps,
            "outer_opt_type":   self.cfg.outer_optimizer.type,
            "total_tokens":     self.token_counter.total_tokens,
            "comm_bytes":       self.comm_tracker.total_bytes,
            "comm_summary":     self.comm_tracker.summary(),
            **{f"final_{k}": v for k, v in final_metrics.items()},
        }
