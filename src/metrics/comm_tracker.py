"""
Communication tracker.

Records actual bytes sent during each outer synchronization so the paper's
communication-quality Pareto frontier (E6) is based on measured data, not
theoretical parameter counts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn


@dataclass
class SyncEvent:
    outer_step: int
    total_tokens: int
    payload_bytes: int       # bytes sent in this synchronization
    duration_ms: float       # wall-clock time spent


class CommunicationTracker:
    """
    Tracks synchronization payload and timing for DiLoCo-style outer steps.

    Usage::

        tracker = CommunicationTracker()
        with tracker.record(outer_step=t, total_tokens=tok):
            # ... do the allreduce / averaging ...
    """

    def __init__(self) -> None:
        self._events: List[SyncEvent] = []
        self._total_bytes: int        = 0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    class _Record:
        def __init__(self, tracker: "CommunicationTracker", outer_step: int, total_tokens: int):
            self._tracker      = tracker
            self._outer_step   = outer_step
            self._total_tokens = total_tokens
            self._start: float = 0.0
            self._bytes: int   = 0

        def set_payload(self, model: nn.Module) -> None:
            """Call inside the context to register model-parameter payload."""
            self._bytes = sum(
                p.numel() * p.element_size()
                for p in model.parameters()
                if p.requires_grad
            )

        def set_bytes(self, n_bytes: int) -> None:
            """Manually set payload size (e.g., for quantized or partial sync)."""
            self._bytes = n_bytes

        def __enter__(self):
            self._start = time.perf_counter()
            return self

        def __exit__(self, *_):
            duration_ms = (time.perf_counter() - self._start) * 1000.0
            event = SyncEvent(
                outer_step=self._outer_step,
                total_tokens=self._total_tokens,
                payload_bytes=self._bytes,
                duration_ms=duration_ms,
            )
            self._tracker._events.append(event)
            self._tracker._total_bytes += self._bytes

    def record(self, outer_step: int, total_tokens: int) -> "_Record":
        return self._Record(self, outer_step, total_tokens)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def total_gb(self) -> float:
        return self._total_bytes / (1024 ** 3)

    def bytes_per_token(self, total_training_tokens: int) -> float:
        if total_training_tokens == 0:
            return 0.0
        return self._total_bytes / total_training_tokens

    def events(self) -> List[SyncEvent]:
        return list(self._events)

    def summary(self) -> dict:
        if not self._events:
            return {"n_syncs": 0, "total_bytes": 0, "avg_duration_ms": 0.0}
        durations = [e.duration_ms for e in self._events]
        return {
            "n_syncs":         len(self._events),
            "total_bytes":     self._total_bytes,
            "total_gb":        round(self.total_gb, 4),
            "avg_duration_ms": round(sum(durations) / len(durations), 2),
            "max_duration_ms": round(max(durations), 2),
        }
