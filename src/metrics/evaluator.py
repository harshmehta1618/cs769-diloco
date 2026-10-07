"""
Evaluator.

Computes validation cross-entropy (primary metric) at fixed token budgets.
Also computes perplexity and, optionally, zero-shot downstream accuracy
via lm-evaluation-harness.

Primary endpoint:   validation cross-entropy at the *final* token budget
                    (not the best-checkpoint loss).
"""

from __future__ import annotations

import math
from typing import Dict, Iterator, List, Optional

import torch
import torch.nn as nn
from torch import Tensor


class Evaluator:
    """
    Runs validation on a held-out token stream.

    Parameters
    ----------
    eval_data_iter  : iterator that yields batches (B, seq_len+1)
    n_eval_tokens   : total validation tokens to consume per eval call
    device          : torch device
    """

    def __init__(
        self,
        eval_data_iter: Iterator,
        n_eval_tokens: int,
        device: torch.device,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.float32,
    ) -> None:
        self._iter          = eval_data_iter
        self.n_eval_tokens  = n_eval_tokens
        self._device        = device
        self.use_amp        = use_amp
        self.amp_dtype      = amp_dtype

    def evaluate(self, model: nn.Module) -> dict:
        """
        Compute validation cross-entropy and perplexity.

        Returns
        -------
        dict with keys: val_loss, val_ppl, val_tokens_evaluated
        """
        model.eval()
        total_loss, total_tokens = 0.0, 0

        with torch.no_grad():
            for batch in self._iter:
                batch = batch.to(self._device)
                input_ids = batch[:, :-1].contiguous()
                targets   = batch[:, 1:].contiguous()
                with torch.autocast(device_type=self._device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    _, loss = model(input_ids, targets)
                n_tok     = targets.numel()
                total_loss   += loss.item() * n_tok
                total_tokens += n_tok
                if total_tokens >= self.n_eval_tokens:
                    break

        model.train()

        if total_tokens == 0:
            return {"val_loss": float("nan"), "val_ppl": float("nan"), "val_tokens": 0}

        val_loss = total_loss / total_tokens
        val_ppl  = math.exp(min(val_loss, 20))   # cap at exp(20) to avoid overflow
        return {
            "val_loss":   val_loss,
            "val_ppl":    val_ppl,
            "val_tokens": total_tokens,
        }

    def evaluate_downstream(
        self,
        model: nn.Module,
        tasks: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate on downstream zero-shot tasks (HellaSwag, PIQA, ARC-Easy).
        Uses lm-evaluation-harness if available; gracefully logs warning otherwise.
        """
        if not tasks:
            return {}

        results: Dict[str, float] = {}
        try:
            import lm_eval  # type: ignore
            # lm-evaluation-harness API
            task_dict = lm_eval.evaluator.simple_evaluate(
                model="hf",
                model_args=model,
                tasks=tasks,
                device=str(self._device),
                batch_size="auto",
            )
            for t in tasks:
                if t in task_dict.get("results", {}):
                    acc = task_dict["results"][t].get("acc,none", float("nan"))
                    results[f"downstream_{t}_acc"] = acc
        except ImportError:
            for t in tasks:
                results[f"downstream_{t}_acc"] = float("nan")
        except Exception:
            for t in tasks:
                results[f"downstream_{t}_acc"] = float("nan")
        return results


# ---------------------------------------------------------------------------
# SNOO Recovery Fraction (E2)
# ---------------------------------------------------------------------------

def snoo_recovery_fraction(
    loss_dp:    float,
    loss_snoo:  float,
    loss_diloco: float,
) -> Optional[float]:
    """
    (Loss_DP − Loss_SNOO) / (Loss_DP − Loss_DiLoCo)

    Returns None when the denominator is near-zero (DiLoCo not better than DP).
    """
    denom = loss_dp - loss_diloco
    if abs(denom) < 1e-6:
        return None
    return (loss_dp - loss_snoo) / denom


# ---------------------------------------------------------------------------
# Gradient noise proxy (E4)
# ---------------------------------------------------------------------------

def gradient_norm_variance(
    model: nn.Module,
    data_iter: Iterator,
    n_microbatches: int,
    device: torch.device,
) -> float:
    """
    Lightweight proxy for gradient noise scale.
    Computes the variance of gradient-L2-norms across n_microbatches.
    """
    model.train()
    norms: List[float] = []

    for _ in range(n_microbatches):
        batch = next(data_iter).to(device)
        input_ids = batch[:, :-1].contiguous()
        targets   = batch[:, 1:].contiguous()
        model.zero_grad()
        _, loss = model(input_ids, targets)
        loss.backward()
        grad_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in model.parameters()
            if p.grad is not None
        ) ** 0.5
        norms.append(grad_norm)
        model.zero_grad()

    mean_norm = sum(norms) / len(norms)
    variance  = sum((n - mean_norm) ** 2 for n in norms) / len(norms)
    return variance
