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
import io

# Force UTF-8 output on Windows
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

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
    # Default model seq_len to data seq_len if not explicitly overridden
    if mc.seq_len == 0 and cfg.data.seq_len > 0:
        base.seq_len = cfg.data.seq_len
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
    parser.add_argument("--dry_run", action="store_true",
                        help="Execute a minimal 1-2 step verification run to validate pipeline & memory")
    args = parser.parse_args()

    # Load + override config
    cfg = load_config(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    if args.dry_run:
        logger.info("[DRY-RUN] MODE ACTIVATED: executing 1-2 steps to verify pipeline, GPU memory, and checkpointing.")
        cfg.distribution.local_batch_size = min(cfg.distribution.local_batch_size, 4)
        cfg.data.seq_len = min(cfg.data.seq_len, 128)
        cfg.data.num_workers = 0
        step_tokens = max(cfg.distribution.local_batch_size * cfg.data.seq_len, 128)
        cfg.data.token_budget = step_tokens * max(cfg.distribution.M_workers, 1) * 2
        cfg.data.eval_token_budget = step_tokens
        cfg.eval.eval_every_tokens = 1
        cfg.eval.checkpoint_every_tokens = 1
        cfg.eval.use_wandb = False
        if cfg.distribution.H_inner_steps > 2:
            cfg.distribution.H_inner_steps = 2
        if cfg.distribution.M_workers > 2:
            cfg.distribution.M_workers = 2
        is_cpu = (cfg.runtime.device == "cpu") or not torch.cuda.is_available()
        if is_cpu and cfg.model.preset in ("160m", "410m", "1b"):
            logger.info("[DRY-RUN] Running on CPU: testing with '60m' architecture to match CPU memory constraints.")
            cfg.model.preset = "60m"
        try:
            import datasets  # type: ignore
        except ImportError:
            if cfg.data.corpus != "synthetic":
                logger.info("[DRY-RUN] 'datasets' package not found in current environment. Using synthetic tokens for dry-run verification.")
                cfg.data.corpus = "synthetic"

    logger.info(f"Run: {cfg.run_name}  |  method: {cfg.distribution.method}")
    logger.info(f"Token budget: {cfg.data.token_budget:,}")

    # Reproducibility
    set_seed(cfg.runtime.seed)

    # Device
    device = resolve_device(cfg.runtime.device)
    logger.info(f"Device: {device}")

    logger.info(f"Precision: {cfg.runtime.precision} (handled via torch.autocast)")

    # Model
    model = build_model_from_cfg(cfg)
    if cfg.runtime.compile and hasattr(torch, "compile"):
        if device.type == "cpu":
            logger.info("torch.compile: skipping on CPU to avoid host C++ compiler dependency.")
        else:
            try:
                model = torch.compile(model)
                logger.info("Model compiled with torch.compile()")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}. Running in standard eager mode.")

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

    # Downstream evaluation (if requested)
    if cfg.eval.downstream_tasks:
        logger.info(f"Evaluating downstream zero-shot tasks: {cfg.eval.downstream_tasks}")
        downstream_metrics = trainer.evaluator.evaluate_downstream(trainer.model, cfg.eval.downstream_tasks)
        for t_k, t_v in downstream_metrics.items():
            logger.info(f"  {t_k}: {t_v}")
        results.update(downstream_metrics)

    logger.info("=" * 60)
    logger.info("Training complete.")
    for k, v in results.items():
        if not isinstance(v, dict):
            logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("[DRY-RUN] SUCCESSFUL: verified model initialization, forward pass, backward pass, checkpointing, and evaluation.")


if __name__ == "__main__":
    main()
