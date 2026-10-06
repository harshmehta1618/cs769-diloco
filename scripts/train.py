"""
Main training entry point.

Usage::

    # Train with a config file
    python scripts/train.py --config configs/e2_attribution_snoo.yaml

    # Override individual fields
    python scripts/train.py --config configs/pilot_60m.yaml \\
        --distribution.method diloco \\
        --distribution.M_workers 4 \\
        --distribution.H_inner_steps 100 \\
        --runtime.seed 1

All overrides follow the dotted config path (e.g. model.n_layers, data.corpus).
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from pathlib import Path

import torch
import numpy as np

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import build_worker_datasets, make_dataloader
from src.models.gpt import build_model, GPTConfig, PRESET_CONFIGS
from src.trainers import build_trainer
from src.utils.config import RunConfig, load_config, save_config
from src.utils.logging_utils import get_logger

logger = get_logger("train")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def build_model_from_cfg(cfg: RunConfig) -> torch.nn.Module:
    mc = cfg.model
    if mc.preset in PRESET_CONFIGS:
        base = copy.deepcopy(PRESET_CONFIGS[mc.preset])
    else:
        base = GPTConfig()

    # Apply per-field overrides
    for attr in ("n_layers", "d_model", "n_heads", "seq_len", "vocab_size"):
        val = getattr(mc, attr)
        if val > 0:
            setattr(base, attr, val)
    base.dropout = mc.dropout
    base.bias    = mc.bias
    base.__post_init__()

    model = build_model(base)
    logger.info(
        f"Model: {mc.preset} | "
        f"params={model.num_parameters():,} | "
        f"n_layers={base.n_layers} d_model={base.d_model} n_heads={base.n_heads}"
    )
    return model


def apply_overrides(cfg: RunConfig, overrides: list[str]) -> RunConfig:
    """
    Apply dotted-path key=value overrides to a RunConfig.
    Example:  ["distribution.M_workers=4", "runtime.seed=1"]
    """
    import yaml
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got: {ov!r}")
        key, val = ov.split("=", 1)
        parts = key.strip().split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        field = parts[-1]
        current = getattr(obj, field)
        # Cast to the same type as the current value
        try:
            casted = type(current)(yaml.safe_load(val))
        except Exception:
            casted = val
        setattr(obj, field, casted)
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CS769 DiLoCo/SNOO Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Config overrides in key=value format, e.g. distribution.M_workers=4"
    )
    parser.add_argument("--log_geometry", action="store_true",
                        help="Log worker-diversity geometry metrics (E3; slower)")
    args = parser.parse_args()

    # Load + override config
    cfg = load_config(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    logger.info(f"Run: {cfg.run_name}  |  method: {cfg.distribution.method}")
    logger.info(f"Token budget: {cfg.data.token_budget:,}")

    # Reproducibility
    set_seed(cfg.runtime.seed)

    # Device
    device = resolve_device(cfg.runtime.device)
    logger.info(f"Device: {device}")

    # Mixed precision (if requested)
    if cfg.runtime.precision in ("bf16", "fp16") and device.type == "cuda":
        torch.set_default_dtype(
            torch.bfloat16 if cfg.runtime.precision == "bf16" else torch.float16
        )

    # Model
    model = build_model_from_cfg(cfg)
    if cfg.runtime.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    # Data
    d = cfg.data
    method = cfg.distribution.method
    M      = cfg.distribution.M_workers
    shard_mode = "clone" if method == "clone_diloco" else d.shard_mode

    datasets = build_worker_datasets(
        corpus=d.corpus,
        tokenizer_name=d.tokenizer,
        seq_len=d.seq_len,
        M_workers=M,
        shard_mode=shard_mode,
        shard_seed=d.shard_seed,
        vocab_size=50257,           # GPT-2 vocab (for real tokenizer)
        token_budget=d.token_budget,
    )
    train_iters = {
        w: iter(make_dataloader(datasets[w], cfg.distribution.local_batch_size, d.num_workers))
        for w in range(M)
    }

    # Eval dataset (always worker 0, IID)
    eval_ds = build_worker_datasets(
        corpus=d.corpus,
        tokenizer_name=d.tokenizer,
        seq_len=d.seq_len,
        M_workers=1,
        shard_mode="iid",
        shard_seed=d.shard_seed + 9999,
        vocab_size=50257,
        token_budget=d.eval_token_budget,
    )[0]
    eval_iter = iter(make_dataloader(eval_ds, cfg.distribution.local_batch_size))

    # Save frozen config
    out_dir = Path(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")
    logger.info(f"Config saved -> {out_dir / 'config.yaml'}")

    # Build and run trainer
    trainer = build_trainer(cfg, model, train_iters, eval_iter, device,
                            log_geometry=args.log_geometry)
    results = trainer.train()

    logger.info("=" * 60)
    logger.info("Training complete.")
    for k, v in results.items():
        if not isinstance(v, dict):
            logger.info(f"  {k}: {v}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
