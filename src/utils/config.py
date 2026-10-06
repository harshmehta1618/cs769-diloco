"""
Configuration dataclasses + YAML loading.

Every reported run must have an immutable config file. The Git commit hash
is embedded into the config at load-time so checkpoints are always traceable.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    preset: str = "debug"           # key in PRESET_CONFIGS
    # Overrides (0 = use preset default)
    n_layers: int  = 0
    d_model: int   = 0
    n_heads: int   = 0
    seq_len: int   = 0
    vocab_size: int = 0
    dropout: float = 0.0
    bias: bool     = True


@dataclass
class DataConfig:
    corpus: str           = "synthetic"   # "c4" | "dolma" | "fineweb" | "synthetic"
    tokenizer: str        = "gpt2"
    token_budget: int     = 1_000_000     # total training tokens
    eval_token_budget: int = 100_000      # tokens for validation
    shard_mode: str       = "iid"         # "iid" | "domain_skew" | "clone"
    shard_seed: int       = 42
    seq_len: int          = 128
    num_workers: int      = 0             # DataLoader workers


@dataclass
class InnerOptimizerConfig:
    lr: float           = 1e-3
    betas: List[float]  = field(default_factory=lambda: [0.9, 0.95])
    eps: float          = 1e-8
    weight_decay: float = 0.1
    grad_clip: float    = 1.0
    warmup_tokens: int  = 0             # 0 = no warmup
    min_lr_ratio: float = 0.1           # cosine decay floor


@dataclass
class OuterOptimizerConfig:
    type: str       = "none"    # "none" | "nesterov" | "avg"
    outer_lr: float = 0.7
    momentum: float = 0.9
    nesterov: bool  = True
    reset_inner: bool = True    # reset inner-optimizer state after each outer step


@dataclass
class DistributionConfig:
    method: str         = "dp_adamw"   # "dp_adamw" | "sw_adamw" | "snoo" | "periodic_avg" | "diloco" | "clone_diloco"
    M_workers: int      = 1            # number of workers / replicas
    H_inner_steps: int  = 1            # inner steps per outer step (H)
    local_batch_size: int = 32
    grad_accumulation: int = 1         # gradient accumulation steps


@dataclass
class RuntimeConfig:
    seed: int                   = 0
    device: str                 = "cpu"    # "cpu" | "cuda" | "cuda:0" | …
    precision: str              = "fp32"   # "fp32" | "bf16" | "fp16"
    compile: bool               = False    # torch.compile
    distributed_backend: str    = "gloo"   # "gloo" | "nccl"


@dataclass
class EvalConfig:
    eval_every_tokens: int      = 50_000
    log_every_steps: int        = 10
    checkpoint_every_tokens: int = 200_000
    use_wandb: bool             = False
    wandb_project: str          = "cs769-diloco"
    downstream_tasks: List[str] = field(default_factory=list)   # e.g. ["hellaswag"]


@dataclass
class RunConfig:
    """Top-level config that describes one complete training run."""
    run_name: str                 = "unnamed"
    experiment_id: str            = "E0"
    model: ModelConfig            = field(default_factory=ModelConfig)
    data: DataConfig              = field(default_factory=DataConfig)
    inner_optimizer: InnerOptimizerConfig = field(default_factory=InnerOptimizerConfig)
    outer_optimizer: OuterOptimizerConfig = field(default_factory=OuterOptimizerConfig)
    distribution: DistributionConfig      = field(default_factory=DistributionConfig)
    runtime: RuntimeConfig                = field(default_factory=RuntimeConfig)
    eval: EvalConfig                      = field(default_factory=EvalConfig)
    # Populated at runtime
    git_sha: str = "unknown"
    output_dir: str = "runs"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _dict_to_dataclass(dc_cls, d: dict):
    """Recursively convert a nested dict to a dataclass, ignoring unknown keys."""
    if not isinstance(d, dict):
        return d
    field_types = {f.name: f.type for f in dc_cls.__dataclass_fields__.values()}  # type: ignore
    kwargs = {}
    for key, val in d.items():
        if key not in field_types:
            continue
        hint = field_types[key]
        # Resolve string type hints to actual classes
        if isinstance(hint, str):
            import sys
            frame = sys._getframe(1)
            hint = eval(hint, frame.f_globals, frame.f_locals)
        if hasattr(hint, "__dataclass_fields__") and isinstance(val, dict):
            kwargs[key] = _dict_to_dataclass(hint, val)
        else:
            kwargs[key] = val
    return dc_cls(**{**asdict(dc_cls()), **kwargs})


def load_config(path: str | Path) -> RunConfig:
    """Load a YAML config file and return a validated RunConfig."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = RunConfig(
        run_name=raw.get("run_name", path.stem),
        experiment_id=raw.get("experiment_id", "E0"),
        model=_dict_to_dataclass(ModelConfig, raw.get("model", {})),
        data=_dict_to_dataclass(DataConfig, raw.get("data", {})),
        inner_optimizer=_dict_to_dataclass(InnerOptimizerConfig, raw.get("inner_optimizer", {})),
        outer_optimizer=_dict_to_dataclass(OuterOptimizerConfig, raw.get("outer_optimizer", {})),
        distribution=_dict_to_dataclass(DistributionConfig, raw.get("distribution", {})),
        runtime=_dict_to_dataclass(RuntimeConfig, raw.get("runtime", {})),
        eval=_dict_to_dataclass(EvalConfig, raw.get("eval", {})),
        output_dir=raw.get("output_dir", "runs"),
    )
    cfg.git_sha = _get_git_sha()
    return cfg


def save_config(cfg: RunConfig, path: str | Path) -> None:
    """Persist config as YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False, sort_keys=False)
