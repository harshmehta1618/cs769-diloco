# CS769 Project: Where Does the Free Lunch Come From?
## Disentangling Outer Momentum from Worker Diversity in Low-Communication LLM Training

---

## Repository Structure

```
cs769project/
├── configs/                        # Immutable YAML configs per experiment
│   ├── debug_smoke.yaml            # Smoke test config (tiny model)
│   ├── e0_harness.yaml             # E0 — harness parity (Gate 1)
│   ├── e1_pilot_60m.yaml           # E1 — 60M pilot validation
│   ├── e2_attribution_160m.yaml    # E2 — main causal decomposition
│   ├── e3_horizon_h.yaml           # E3 — inner step horizon H sweep
│   ├── e4_worker_scaling.yaml      # E4 — worker count M scaling
│   └── e5_clone_diloco.yaml        # E5 — Clone-DiLoCo data diversity ablation
├── src/
│   ├── models/
│   │   └── gpt.py                  # GPT-style LM (presets: debug/60m/160m/410m/1b)
│   ├── data/
│   │   └── dataset.py              # Synthetic + C4/Dolma, IID/clone/skew shards
│   ├── optimizers/
│   │   ├── inner.py                # AdamW inner optimizer + token-driven LR
│   │   └── outer.py                # Nesterov (DiLoCo/SNOO) + AvgOuter (PeriodicAvg)
│   ├── trainers/
│   │   ├── base_trainer.py         # Abstract base: token counter, eval, checkpoint
│   │   ├── dp_adamw.py             # DP-AdamW baseline
│   │   ├── two_loop_trainer.py     # SNOO / PeriodicAvg / DiLoCo (unified)
│   │   └── __init__.py             # build_trainer() factory
│   ├── metrics/
│   │   ├── comm_tracker.py         # Actual comm bytes + timing per outer step
│   │   ├── geometry.py             # Pseudo-grad cosine sim, worker distance (E3/E6)
│   │   └── evaluator.py            # Val loss, perplexity, SNOO recovery fraction
│   └── utils/
│       ├── config.py               # RunConfig dataclasses + YAML load/save
│       ├── token_counter.py        # Single token counter (drives LR + eval)
│       ├── logging_utils.py        # Console + JSONL summary + W&B
│       └── checkpoint.py           # Save/load: model + both optimizers + token counter
├── scripts/
│   ├── train.py                    # Unified entry point for all experiments
│   └── run_experiments.py          # Batch runner across experiment suites and seeds
├── analysis/
│   ├── plot_attribution.py         # Fig 3: Δloss bar plot (E2)
│   ├── plot_scaling.py             # Fig 4 & 5: Horizon H & Worker M scaling
│   ├── plot_geometry.py            # Fig 6: Trajectory geometry & consensus metrics
│   └── compute_snoo_frac.py        # SNOO recovery fraction computation (E2)
├── smoke_test.py                   # 14-test suite — all methods in < 5s
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run smoke tests (validates all 5 methods + metrics + checkpoint)
python smoke_test.py

# 3. Harness parity check (E0 Gate 1)
python scripts/train.py --config configs/e0_harness.yaml

# 4. Main causal decomposition (E2) — change method via override
python scripts/train.py \
    --config configs/e2_attribution_160m.yaml \
    --override distribution.method=diloco runtime.seed=0

python scripts/train.py \
    --config configs/e2_attribution_160m.yaml \
    --override distribution.method=snoo runtime.seed=0

python scripts/train.py \
    --config configs/e2_attribution_160m.yaml \
    --override distribution.method=dp_adamw runtime.seed=0
```

---

## Training Methods

| Method flag | Algorithm | Workers | Outer update |
|-------------|-----------|---------|-------------|
| `dp_adamw` | Standard DDP AdamW | M (sync) | None |
| `sw_adamw` | Single-worker AdamW | 1 | None |
| `snoo` | SNOO | 1 | Nesterov on pseudo-grad |
| `periodic_avg` | PeriodicAvg / FedAvg | M | Plain averaging |
| `diloco` | DiLoCo | M | Nesterov on avg pseudo-grad |
| `clone_diloco` | Clone-DiLoCo | M (identical data) | Nesterov |

---

## Key Design Principles

1. **One codebase, all methods** — algorithm differences selected by config
2. **Token counter as universal clock** — not optimizer steps (H changes meaning)
3. **Primary metric = final token budget checkpoint** — not best-validation checkpoint
4. **Actual comm bytes tracked** — not theoretical parameter counts
5. **Immutable config + git SHA in every checkpoint** — all results traceable

---

## Experiments (from PDF)

| ID | Scale | Command |
|----|-------|---------|
| E0 | 60M | `--config configs/e0_harness.yaml` |
| E1 | 160M | `--override distribution.method=diloco` |
| E2 | 160M | `--config configs/e2_attribution_160m.yaml` + method sweep |
| E3 | 160M | `--override data.shard_mode=clone --log_geometry` |
| E4 | 160M | `--override distribution.grad_accumulation=4` |

---

## References

1. Charles et al. (2025). *Scaling Laws for DiLoCo.* arXiv:2503.09799
2. Douillard et al. (2024). *DiLoCo.* arXiv:2311.08105
3. Kallusky et al. (2025). *SNOO.* arXiv:2510.15830
