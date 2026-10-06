"""
Trainer factory and package init.

Usage::

    from src.trainers import build_trainer

    trainer = build_trainer(cfg, model, train_iters, eval_iter, device)
    results = trainer.train()
"""

from __future__ import annotations

from typing import Dict, Iterator

import torch
import torch.nn as nn

from .base_trainer import BaseTrainer
from .dp_adamw import DPAdamWTrainer
from .two_loop_trainer import TwoLoopTrainer
from ..utils.config import RunConfig


def build_trainer(
    cfg: RunConfig,
    model: nn.Module,
    train_iters: Dict[int, Iterator],
    eval_iter: Iterator,
    device: torch.device,
    log_geometry: bool = False,
) -> BaseTrainer:
    """
    Return the appropriate trainer for cfg.distribution.method.

    Methods mapping:
      "dp_adamw"    → DPAdamWTrainer
      "sw_adamw"    → DPAdamWTrainer (M=1, no outer opt)
      "snoo"        → TwoLoopTrainer (M=1, outer=nesterov)
      "periodic_avg"→ TwoLoopTrainer (M≥1, outer=avg)
      "diloco"      → TwoLoopTrainer (M≥1, outer=nesterov)
      "clone_diloco"→ TwoLoopTrainer (shard_mode=clone, outer=nesterov)
    """
    method = cfg.distribution.method

    if method in ("dp_adamw", "sw_adamw"):
        return DPAdamWTrainer(cfg, model, train_iters, eval_iter, device)

    if method in ("snoo", "periodic_avg", "diloco", "clone_diloco"):
        return TwoLoopTrainer(
            cfg, model, train_iters, eval_iter, device,
            log_geometry=log_geometry,
        )

    raise ValueError(
        f"Unknown training method: {method!r}. "
        f"Choose from: dp_adamw, sw_adamw, snoo, periodic_avg, diloco, clone_diloco"
    )


__all__ = [
    "BaseTrainer",
    "DPAdamWTrainer",
    "TwoLoopTrainer",
    "build_trainer",
]
