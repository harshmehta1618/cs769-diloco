"""
DP-AdamW Trainer (Experiment E0 / E1 baseline).

Standard synchronous data-parallel AdamW. On a single machine this is
the plain single-process version; for multi-GPU the caller should wrap
the model in torch.nn.parallel.DistributedDataParallel before passing it.

Every gradient step advances the token counter, which drives the LR schedule
and evaluation triggers — identical to all other trainers.
"""

from __future__ import annotations

from typing import Dict, Iterator

import torch
import torch.nn as nn

from .base_trainer import BaseTrainer
from ..utils.config import RunConfig


class DPAdamWTrainer(BaseTrainer):
    """
    Gold-standard synchronous data-parallel AdamW baseline.

    This trainer processes one device's share of the global batch each step.
    If running on multiple devices via DDP, the all-reduce is handled
    transparently by PyTorch; the token counter counts global tokens.
    """

    def __init__(
        self,
        cfg: RunConfig,
        model: nn.Module,
        train_iters: Dict[int, Iterator],
        eval_iter: Iterator,
        device: torch.device,
    ) -> None:
        super().__init__(cfg, model, train_iters, eval_iter, device)

    def train(self) -> dict:
        """
        Main training loop.
        Runs until cfg.data.token_budget is exhausted.
        Returns a dict of final metrics.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        step = 0
        accum_steps = self.cfg.distribution.grad_accumulation

        while not self.token_counter.budget_exhausted:
            loss_accum = 0.0

            # Gradient accumulation loop
            for micro_step in range(accum_steps):
                input_ids, targets = self._next_batch(worker_idx=0)
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    _, loss = self.model(input_ids, targets)
                    loss    = loss / accum_steps

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                loss_accum += loss.item()
                n_tokens = self._count_tokens(targets)
                self.token_counter.step(n_tokens)

            # LR update (token-driven)
            lr = self._update_lr()

            # Gradient clip + step
            grad_norm = self._clip_and_step()

            step += 1
            self._global_step = step

            # Logging
            self.logger.log(
                step=step,
                total_tokens=self.token_counter.total_tokens,
                train_loss=loss_accum,
                lr=lr,
                grad_norm=grad_norm,
            )

            # Evaluation
            eval_metrics = self._maybe_evaluate()
            if eval_metrics:
                self.logger.log(
                    step=step,
                    total_tokens=self.token_counter.total_tokens,
                    **eval_metrics,
                )

            # Checkpoint
            self._maybe_checkpoint()

            if self.token_counter.budget_exhausted:
                break

        # Final evaluation at end of training (primary result)
        final_metrics = self.evaluator.evaluate(self.model)
        self.logger.log(
            step=step,
            total_tokens=self.token_counter.total_tokens,
            **{f"final_{k}": v for k, v in final_metrics.items()},
        )
        self.logger.finish()

        return {
            "method":       "dp_adamw",
            "total_tokens": self.token_counter.total_tokens,
            "comm_bytes":   self.comm_tracker.total_bytes,   # 0 for DP (no outer sync)
            **{f"final_{k}": v for k, v in final_metrics.items()},
        }
