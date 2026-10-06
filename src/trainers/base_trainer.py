"""
Abstract base trainer.

All concrete trainers (DP-AdamW, SW-AdamW, SNOO, PeriodicAvg, DiLoCo)
inherit from this class and implement `_train_step` and `train`.
"""

from __future__ import annotations

import copy
import random
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterator, Optional

import torch
import torch.nn as nn

from ..data.dataset import unpack_batch
from ..metrics.comm_tracker import CommunicationTracker
from ..metrics.evaluator import Evaluator
from ..optimizers.inner import build_inner_optimizer, update_lr
from ..utils.checkpoint import save_checkpoint, latest_checkpoint
from ..utils.config import RunConfig
from ..utils.logging_utils import TrainingLogger
from ..utils.token_counter import TokenCounter


class BaseTrainer(ABC):
    """
    Abstract trainer providing the shared training loop scaffolding:
      - token counter (single source of truth for LR and eval triggers)
      - inner optimizer construction
      - LR scheduling (cosine decay with linear warmup, keyed on tokens)
      - checkpoint save/resume
      - evaluation dispatch
      - logging
    """

    def __init__(
        self,
        cfg: RunConfig,
        model: nn.Module,
        train_iters: Dict[int, Iterator],   # {worker_idx: data_iterator}
        eval_iter: Iterator,
        device: torch.device,
    ) -> None:
        self.cfg    = cfg
        self.model  = model.to(device)
        self.device = device

        self.train_iters = train_iters
        self.eval_iter   = eval_iter

        # Single unified token counter drives LR and eval
        self.token_counter = TokenCounter(cfg.data.token_budget)

        # Inner optimizer
        io = cfg.inner_optimizer
        self.optimizer = build_inner_optimizer(
            model=self.model,
            lr=io.lr,
            betas=io.betas,
            eps=io.eps,
            weight_decay=io.weight_decay,
        )

        # Communication tracker (tracks outer-sync payload)
        self.comm_tracker = CommunicationTracker()

        # Evaluator
        self.evaluator = Evaluator(
            eval_data_iter=eval_iter,
            n_eval_tokens=cfg.data.eval_token_budget,
            device=device,
        )

        # Output
        self.output_dir = Path(cfg.output_dir) / cfg.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Logger
        self.logger = TrainingLogger(
            run_name=cfg.run_name,
            output_dir=self.output_dir,
            use_wandb=cfg.eval.use_wandb,
            wandb_project=cfg.eval.wandb_project,
            config_dict=asdict(cfg),
        )
        self.logger.set_log_every(cfg.eval.log_every_steps)

        self._global_step = 0

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    @abstractmethod
    def train(self) -> dict:
        """
        Run the full training loop until token_budget is exhausted.
        Returns a dict of final metrics.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _update_lr(self) -> float:
        io = self.cfg.inner_optimizer
        return update_lr(
            optimizer=self.optimizer,
            total_tokens=self.token_counter.total_tokens,
            token_budget=self.token_counter.token_budget,
            peak_lr=io.lr,
            warmup_tokens=io.warmup_tokens,
            min_lr_ratio=io.min_lr_ratio,
        )

    def _clip_and_step(self) -> float:
        """Gradient clipping + inner optimizer step. Returns grad norm."""
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.inner_optimizer.grad_clip
        ).item()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return grad_norm

    def _maybe_evaluate(self) -> Optional[dict]:
        """Evaluate if the eval trigger has been reached."""
        freq = self.cfg.eval.eval_every_tokens
        if (self.token_counter.total_tokens % freq) < (
            self.cfg.distribution.local_batch_size * self.cfg.data.seq_len
        ):
            return self.evaluator.evaluate(self.model)
        return None

    def _maybe_checkpoint(self, outer_opt_state: Optional[dict] = None) -> None:
        freq = self.cfg.eval.checkpoint_every_tokens
        if (self.token_counter.total_tokens % freq) < (
            self.cfg.distribution.local_batch_size * self.cfg.data.seq_len
        ):
            save_checkpoint(
                output_dir=self.output_dir,
                step=self._global_step,
                total_tokens=self.token_counter.total_tokens,
                model=self.model,
                inner_optimizer=self.optimizer,
                token_counter_state=self.token_counter.state_dict(),
                config_dict=asdict(self.cfg),
                outer_optimizer_state=outer_opt_state,
                git_sha=self.cfg.git_sha,
            )

    def _next_batch(self, worker_idx: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        batch = next(self.train_iters[worker_idx])
        return unpack_batch(batch, self.device)

    def _count_tokens(self, targets: torch.Tensor) -> int:
        return targets.numel()
