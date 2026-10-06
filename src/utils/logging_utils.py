"""
Logging utilities.

Supports:
  - Console logging (via Python logging)
  - JSON run-summary file (for paper tables — never copy from dashboards)
  - Optional Weights & Biases integration
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Console logger
# ---------------------------------------------------------------------------

def get_logger(name: str = "cs769") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


_LOGGER = get_logger()


# ---------------------------------------------------------------------------
# JSON run-summary logger
# ---------------------------------------------------------------------------

class RunSummary:
    """
    Appends log entries as JSONL rows to <output_dir>/run_summary.jsonl.
    Also maintains an in-memory list for quick analysis.
    """

    def __init__(self, output_dir: str | Path, run_name: str) -> None:
        self._dir      = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path     = self._dir / "run_summary.jsonl"
        self._run_name = run_name
        self._rows: list[dict] = []

    def log(self, step: int, total_tokens: int, metrics: dict[str, Any]) -> None:
        row = {
            "run":           self._run_name,
            "step":          step,
            "total_tokens":  total_tokens,
            "wall_time":     time.time(),
            **metrics,
        }
        self._rows.append(row)
        with open(self._path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def flush(self) -> None:
        pass  # JSONL is written immediately

    @property
    def rows(self) -> list[dict]:
        return self._rows

    def to_csv(self, path: Optional[str | Path] = None) -> Path:
        import csv
        out = Path(path) if path else self._dir / "run_summary.csv"
        if not self._rows:
            return out
        # Collect all possible fieldnames (rows may have different columns when eval fires)
        all_keys: list[str] = []
        seen: set[str] = set()
        for row in self._rows:
            for k in row:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            for row in self._rows:
                writer.writerow({k: row.get(k, "") for k in all_keys})
        return out


# ---------------------------------------------------------------------------
# Weights & Biases (optional)
# ---------------------------------------------------------------------------

class WandbLogger:
    """Thin wrapper around wandb — gracefully no-ops if wandb is not installed."""

    def __init__(
        self,
        project: str,
        name: str,
        config: dict,
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        if not enabled:
            return
        try:
            import wandb  # noqa: F401
            self._wandb = wandb
            wandb.init(project=project, name=name, config=config, resume="allow")
        except Exception as e:
            _LOGGER.warning(f"WanDB init failed ({e}); logging disabled.")
            self._enabled = False

    def log(self, metrics: dict, step: int) -> None:
        if not self._enabled:
            return
        try:
            self._wandb.log(metrics, step=step)
        except Exception:
            pass

    def finish(self) -> None:
        if self._enabled:
            try:
                self._wandb.finish()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Composite logger used by trainers
# ---------------------------------------------------------------------------

class TrainingLogger:
    """
    Combines console, JSON summary, and optional W&B into one interface.

    Usage::

        logger = TrainingLogger(cfg, output_dir="runs/my_run")
        logger.log(step=10, total_tokens=5120, loss=2.34, lr=1e-3, ...)
    """

    def __init__(
        self,
        run_name: str,
        output_dir: str | Path,
        use_wandb: bool = False,
        wandb_project: str = "cs769-diloco",
        config_dict: Optional[dict] = None,
    ) -> None:
        self._console = get_logger(run_name)
        self._summary = RunSummary(output_dir, run_name)
        self._wandb   = WandbLogger(
            project=wandb_project,
            name=run_name,
            config=config_dict or {},
            enabled=use_wandb,
        )
        self._log_every = 1

    def set_log_every(self, n: int) -> None:
        self._log_every = max(1, n)

    def log(self, step: int, total_tokens: int, **metrics: Any) -> None:
        self._summary.log(step, total_tokens, metrics)
        self._wandb.log(metrics, step=step)
        if step % self._log_every == 0:
            parts = [f"step={step}", f"tok={total_tokens:,}"]
            for k, v in metrics.items():
                parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
            self._console.info("  ".join(parts))

    def finish(self) -> None:
        self._wandb.finish()
        csv_path = self._summary.to_csv()
        self._console.info(f"Run summary saved -> {csv_path}")
