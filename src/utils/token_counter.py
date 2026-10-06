"""
Unified token counter.

Every training method (DP-AdamW, SNOO, DiLoCo, …) is driven by a single
global token counter so that learning-rate schedules and evaluation triggers
are comparable across algorithms regardless of H or worker count M.
"""

from __future__ import annotations


class TokenCounter:
    """
    Thread-safe-ish running tally of tokens consumed during training.

    Attributes
    ----------
    total_tokens   : int   – total tokens seen since construction (or reset)
    token_budget   : int   – total tokens allowed for this run
    """

    def __init__(self, token_budget: int, initial: int = 0) -> None:
        self.token_budget = int(token_budget)
        self._count: int  = int(initial)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def step(self, n_tokens: int) -> None:
        """Add *n_tokens* to the running total."""
        self._count += int(n_tokens)

    def reset(self) -> None:
        self._count = 0

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def total_tokens(self) -> int:
        return self._count

    @property
    def fraction(self) -> float:
        """Progress in [0, 1]."""
        return self._count / max(self.token_budget, 1)

    @property
    def budget_exhausted(self) -> bool:
        return self._count >= self.token_budget

    @property
    def remaining(self) -> int:
        return max(0, self.token_budget - self._count)

    # ------------------------------------------------------------------
    # LR schedule helpers
    # ------------------------------------------------------------------

    def warmup_fraction(self, warmup_tokens: int) -> float:
        """
        Linear warmup factor in [0, 1].
        Returns 1.0 once warmup_tokens have been seen.
        """
        if warmup_tokens <= 0:
            return 1.0
        return min(1.0, self._count / warmup_tokens)

    def cosine_decay_factor(self, warmup_tokens: int, min_lr_ratio: float = 0.1) -> float:
        """
        Cosine decay factor after warmup, in [min_lr_ratio, 1.0].
        Commonly used as   lr = peak_lr * cosine_decay_factor(...)
        """
        import math
        if self._count < warmup_tokens:
            return self._count / max(warmup_tokens, 1)
        progress = (self._count - warmup_tokens) / max(
            self.token_budget - warmup_tokens, 1
        )
        decay = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return max(min_lr_ratio, decay)

    # ------------------------------------------------------------------
    # Serialisation (for checkpointing)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {"count": self._count, "token_budget": self.token_budget}

    def load_state_dict(self, sd: dict) -> None:
        self._count       = int(sd["count"])
        self.token_budget = int(sd["token_budget"])

    def __repr__(self) -> str:
        return (
            f"TokenCounter({self._count:,}/{self.token_budget:,} tokens, "
            f"{self.fraction * 100:.1f}%)"
        )
