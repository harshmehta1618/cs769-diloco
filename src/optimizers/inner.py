"""
Inner optimizer builder.

All methods share the same AdamW inner optimizer; this module provides a
consistent factory and a learning-rate update function driven by the shared
TokenCounter (not by optimizer steps, which differ across methods).
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


def build_inner_optimizer(
    model: nn.Module,
    lr: float,
    betas: List[float],
    eps: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """
    Build AdamW with separate weight-decay groups.
    Bias, LayerNorm, and embedding parameters are excluded from weight decay
    (standard practice in GPT training).
    """
    decay_params     = []
    no_decay_params  = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    param_groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps, fused=False)


def update_lr(
    optimizer: torch.optim.Optimizer,
    total_tokens: int,
    token_budget: int,
    peak_lr: float,
    warmup_tokens: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """
    Cosine decay with linear warmup, keyed on total token count.
    Returns the new learning rate.
    """
    if total_tokens < warmup_tokens:
        lr = peak_lr * total_tokens / max(warmup_tokens, 1)
    else:
        progress = (total_tokens - warmup_tokens) / max(token_budget - warmup_tokens, 1)
        decay    = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        lr       = peak_lr * max(min_lr_ratio, decay)

    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr
