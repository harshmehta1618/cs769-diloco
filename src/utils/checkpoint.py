"""
Checkpoint utilities.

Saves and loads:
  - Model state_dict
  - Inner optimizer state_dict
  - Outer optimizer state_dict  (for DiLoCo / SNOO)
  - TokenCounter state
  - Run config (YAML snapshot)
  - Git SHA

Layout::

    <output_dir>/
        checkpoints/
            step_<N>/
                model.pt
                inner_optimizer.pt
                outer_optimizer.pt   (optional)
                token_counter.pt
                config.yaml
                meta.json
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch


def save_checkpoint(
    output_dir: str | Path,
    step: int,
    total_tokens: int,
    model: torch.nn.Module,
    inner_optimizer: torch.optim.Optimizer,
    token_counter_state: dict,
    config_dict: dict,
    outer_optimizer_state: Optional[dict] = None,
    git_sha: str = "unknown",
) -> Path:
    """Save a complete training checkpoint."""
    ckpt_dir = Path(output_dir) / "checkpoints" / f"step_{step:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(),          ckpt_dir / "model.pt")
    torch.save(inner_optimizer.state_dict(), ckpt_dir / "inner_optimizer.pt")
    torch.save(token_counter_state,          ckpt_dir / "token_counter.pt")

    if outer_optimizer_state is not None:
        torch.save(outer_optimizer_state, ckpt_dir / "outer_optimizer.pt")

    # Human-readable meta
    meta = {
        "step":           step,
        "total_tokens":   total_tokens,
        "git_sha":        git_sha,
        "saved_at":       time.strftime("%Y-%m-%dT%H:%M:%S"),
        "has_outer_opt":  outer_optimizer_state is not None,
    }
    with open(ckpt_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Config snapshot
    import yaml
    with open(ckpt_dir / "config.yaml", "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    return ckpt_dir


def load_checkpoint(
    ckpt_dir: str | Path,
    model: torch.nn.Module,
    inner_optimizer: torch.optim.Optimizer,
    token_counter,                    # TokenCounter instance
    device: torch.device,
    outer_optimizer=None,             # optional OuterOptimizer instance
) -> dict:
    """
    Load a checkpoint in-place.
    Returns meta dict (step, total_tokens, git_sha, …).
    """
    ckpt_dir = Path(ckpt_dir)
    assert ckpt_dir.exists(), f"Checkpoint directory not found: {ckpt_dir}"

    model.load_state_dict(
        torch.load(ckpt_dir / "model.pt", map_location=device, weights_only=True)
    )
    inner_optimizer.load_state_dict(
        torch.load(ckpt_dir / "inner_optimizer.pt", map_location=device, weights_only=True)
    )
    token_counter.load_state_dict(
        torch.load(ckpt_dir / "token_counter.pt", map_location=device, weights_only=True)
    )

    if outer_optimizer is not None and (ckpt_dir / "outer_optimizer.pt").exists():
        outer_optimizer.load_state_dict(
            torch.load(ckpt_dir / "outer_optimizer.pt", map_location=device, weights_only=True)
        )

    with open(ckpt_dir / "meta.json") as f:
        meta = json.load(f)
    return meta


def latest_checkpoint(output_dir: str | Path) -> Optional[Path]:
    """Return the most recent checkpoint dir, or None."""
    ckpt_root = Path(output_dir) / "checkpoints"
    if not ckpt_root.exists():
        return None
    dirs = sorted(ckpt_root.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    return dirs[-1] if dirs else None
