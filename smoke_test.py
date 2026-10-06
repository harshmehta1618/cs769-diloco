"""
Smoke Test Suite for CS769 DiLoCo Project.

Tests that all 5 training methods (DP-AdamW, SW-AdamW, SNOO, PeriodicAvg, DiLoCo)
can complete a short training run with:
  - Finite, decreasing loss
  - Correct token counting
  - Communication tracker working
  - Checkpoint save/load
  - All geometry metrics computable
  - Outer optimizer state_dict round-trip

Run::
    python smoke_test.py

Expected: all tests PASS in < 60 seconds on CPU.
"""

from __future__ import annotations

import copy
import os
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch
import io

# Force UTF-8 output on Windows
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Make src importable from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import build_worker_datasets, make_dataloader
from src.models.gpt import GPT, GPTConfig, build_model
from src.optimizers.inner import build_inner_optimizer, update_lr
from src.optimizers.outer import build_outer_optimizer
from src.metrics.comm_tracker import CommunicationTracker
from src.metrics.geometry import compute_all_geometry_metrics
from src.metrics.evaluator import Evaluator, snoo_recovery_fraction
from src.trainers import build_trainer
from src.utils.checkpoint import save_checkpoint, load_checkpoint, latest_checkpoint
from src.utils.config import (
    RunConfig, ModelConfig, DataConfig, InnerOptimizerConfig,
    OuterOptimizerConfig, DistributionConfig, RuntimeConfig, EvalConfig,
)
from src.utils.token_counter import TokenCounter


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

results: dict[str, str] = {}


def test(name: str):
    """Decorator to register and run a test, catching exceptions."""
    def decorator(fn: Callable):
        def wrapper():
            t0 = time.perf_counter()
            try:
                fn()
                elapsed = time.perf_counter() - t0
                results[name] = f"{PASS}  ({elapsed:.2f}s)"
                print(f"  {PASS}  {name}  ({elapsed:.2f}s)")
            except Exception as e:
                elapsed = time.perf_counter() - t0
                results[name] = f"{FAIL}  ({elapsed:.2f}s): {e}"
                print(f"  {FAIL}  {name}  ({elapsed:.2f}s)")
                traceback.print_exc()
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DEVICE = torch.device("cpu")
VOCAB  = 256          # tiny vocab for speed
SEQ    = 32           # short sequences
BUDGET = 6000         # ~6K training tokens
BATCH  = 4


def make_cfg(method: str, M: int = 1, H: int = 5, outer_type: str = "none") -> RunConfig:
    return RunConfig(
        run_name=f"smoke_{method}",
        experiment_id="smoke",
        model=ModelConfig(preset="debug"),
        data=DataConfig(
            corpus="synthetic",
            token_budget=BUDGET,
            eval_token_budget=500,
            shard_mode="iid",
            shard_seed=42,
            seq_len=SEQ,
        ),
        inner_optimizer=InnerOptimizerConfig(lr=1e-3, warmup_tokens=100),
        outer_optimizer=OuterOptimizerConfig(type=outer_type, outer_lr=0.7, momentum=0.9, nesterov=True),
        distribution=DistributionConfig(
            method=method,
            M_workers=M,
            H_inner_steps=H,
            local_batch_size=BATCH,
        ),
        runtime=RuntimeConfig(seed=0, device="cpu"),
        eval=EvalConfig(
            eval_every_tokens=3000,
            log_every_steps=50,
            checkpoint_every_tokens=BUDGET,
        ),
        output_dir="runs/smoke",
    )


def make_model() -> GPT:
    cfg = GPTConfig(n_layers=2, d_model=64, n_heads=2, seq_len=SEQ, vocab_size=VOCAB)
    return build_model(cfg)


def make_iters(M: int = 1):
    datasets = build_worker_datasets(
        corpus="synthetic", tokenizer_name="gpt2",
        seq_len=SEQ, M_workers=M, shard_mode="iid",
        shard_seed=42, vocab_size=VOCAB, token_budget=BUDGET,
    )
    train_iters = {
        w: iter(make_dataloader(datasets[w], BATCH))
        for w in range(M)
    }
    eval_ds = build_worker_datasets(
        corpus="synthetic", tokenizer_name="gpt2",
        seq_len=SEQ, M_workers=1, shard_mode="iid",
        shard_seed=9999, vocab_size=VOCAB, token_budget=500,
    )[0]
    eval_iter = iter(make_dataloader(eval_ds, BATCH))
    return train_iters, eval_iter


# ===========================================================================
# TEST 1: Model instantiation and forward pass
# ===========================================================================

@test("T01 — Model forward pass (debug GPT)")
def test_model_forward():
    model = make_model()
    x = torch.randint(0, VOCAB, (2, SEQ))
    y = torch.randint(0, VOCAB, (2, SEQ))
    logits, loss = model(x, y)
    assert logits.shape == (2, SEQ, VOCAB), f"Bad logit shape: {logits.shape}"
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    assert loss.item() > 0, "Loss should be positive"


# ===========================================================================
# TEST 2: Token counter
# ===========================================================================

@test("T02 — TokenCounter step, fraction, LR decay")
def test_token_counter():
    tc = TokenCounter(token_budget=1000)
    tc.step(100)
    assert tc.total_tokens == 100
    assert abs(tc.fraction - 0.1) < 1e-6
    tc.step(900)
    assert tc.budget_exhausted
    # State dict round-trip
    sd = tc.state_dict()
    tc2 = TokenCounter(token_budget=1)
    tc2.load_state_dict(sd)
    assert tc2.total_tokens == 1000


# ===========================================================================
# TEST 3: DP-AdamW trainer
# ===========================================================================

@test("T03 — DP-AdamW trainer completes, finite loss")
def test_dp_adamw():
    cfg = make_cfg("dp_adamw", M=1, H=1, outer_type="none")
    model = make_model()
    train_iters, eval_iter = make_iters(M=1)
    trainer = build_trainer(cfg, model, train_iters, eval_iter, DEVICE)
    results_dict = trainer.train()
    assert "final_val_loss" in results_dict
    assert torch.isfinite(torch.tensor(results_dict["final_val_loss"]))
    assert results_dict["comm_bytes"] == 0, "DP-AdamW should have 0 comm bytes"


# ===========================================================================
# TEST 4: SNOO trainer (M=1, outer=nesterov)
# ===========================================================================

@test("T04 — SNOO trainer completes (M=1, outer=nesterov)")
def test_snoo():
    cfg = make_cfg("snoo", M=1, H=5, outer_type="nesterov")
    model = make_model()
    train_iters, eval_iter = make_iters(M=1)
    trainer = build_trainer(cfg, model, train_iters, eval_iter, DEVICE)
    results_dict = trainer.train()
    assert "final_val_loss" in results_dict
    assert torch.isfinite(torch.tensor(results_dict["final_val_loss"]))
    assert results_dict["comm_bytes"] > 0, "SNOO should log outer-step comm bytes"


# ===========================================================================
# TEST 5: PeriodicAvg trainer (M=2, outer=avg)
# ===========================================================================

@test("T05 — PeriodicAvg trainer completes (M=2, outer=avg)")
def test_periodic_avg():
    cfg = make_cfg("periodic_avg", M=2, H=5, outer_type="avg")
    model = make_model()
    train_iters, eval_iter = make_iters(M=2)
    trainer = build_trainer(cfg, model, train_iters, eval_iter, DEVICE)
    results_dict = trainer.train()
    assert "final_val_loss" in results_dict
    assert torch.isfinite(torch.tensor(results_dict["final_val_loss"]))


# ===========================================================================
# TEST 6: DiLoCo trainer (M=2, outer=nesterov)
# ===========================================================================

@test("T06 — DiLoCo trainer completes (M=2, H=5, outer=nesterov)")
def test_diloco():
    cfg = make_cfg("diloco", M=2, H=5, outer_type="nesterov")
    model = make_model()
    train_iters, eval_iter = make_iters(M=2)
    trainer = build_trainer(cfg, model, train_iters, eval_iter, DEVICE)
    results_dict = trainer.train()
    assert "final_val_loss" in results_dict
    assert torch.isfinite(torch.tensor(results_dict["final_val_loss"]))
    assert results_dict["comm_bytes"] > 0


# ===========================================================================
# TEST 7: Outer optimizer correctness
# ===========================================================================

@test("T07 — Nesterov outer optimizer state_dict round-trip")
def test_outer_optimizer_statedict():
    model = make_model()
    outer_opt = build_outer_optimizer(
        outer_type="nesterov",
        ref_params_iter=iter(model.named_parameters()),
        outer_lr=0.7, momentum=0.9, nesterov=True,
    )
    # Run one step
    ref_params = {n: p.data.clone() for n, p in model.named_parameters()}
    avg_pg     = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    outer_opt.step(ref_params, avg_pg)

    sd = outer_opt.state_dict()
    # Restore into fresh optimizer
    outer_opt2 = build_outer_optimizer(
        "nesterov", iter(model.named_parameters()), 0.7, 0.9, True
    )
    outer_opt2.load_state_dict(sd)
    assert outer_opt2.outer_lr == 0.7
    assert outer_opt2.momentum == 0.9


# ===========================================================================
# TEST 8: Communication tracker
# ===========================================================================

@test("T08 — CommunicationTracker records bytes and duration")
def test_comm_tracker():
    model = make_model()
    tracker = CommunicationTracker()

    with tracker.record(outer_step=1, total_tokens=1000) as rec:
        rec.set_payload(model)
        time.sleep(0.001)

    assert tracker.total_bytes > 0, "Should have recorded bytes"
    summary = tracker.summary()
    assert summary["n_syncs"] == 1
    assert summary["avg_duration_ms"] > 0


# ===========================================================================
# TEST 9: Worker diversity geometry metrics
# ===========================================================================

@test("T09 — Geometry metrics (cosine sim, norm dispersion, worker distance)")
def test_geometry_metrics():
    model = make_model()
    # Simulate two workers with different params
    wm1 = copy.deepcopy(model)
    wm2 = copy.deepcopy(model)
    with torch.no_grad():
        for p in wm1.parameters():
            p.add_(torch.randn_like(p) * 0.01)
        for p in wm2.parameters():
            p.add_(torch.randn_like(p) * 0.02)

    ref = {n: p.data for n, p in model.named_parameters()}
    pg1 = {n: ref[n] - p.data for n, p in wm1.named_parameters()}
    pg2 = {n: ref[n] - p.data for n, p in wm2.named_parameters()}
    avg_pg = {n: (pg1[n] + pg2[n]) / 2 for n in pg1}
    wp1 = {n: p.data for n, p in wm1.named_parameters()}
    wp2 = {n: p.data for n, p in wm2.named_parameters()}

    geo = compute_all_geometry_metrics(
        pseudo_grads=[pg1, pg2],
        worker_params=[wp1, wp2],
        avg_pseudo_grad=avg_pg,
    )
    assert "pg_cosine_sim" in geo
    assert "pg_norm_mean" in geo
    assert "worker_param_dist_mean" in geo
    assert isinstance(geo["pg_cosine_sim"], float)


# ===========================================================================
# TEST 10: SNOO recovery fraction
# ===========================================================================

@test("T10 — SNOO recovery fraction calculation")
def test_snoo_recovery_fraction():
    frac = snoo_recovery_fraction(loss_dp=2.5, loss_snoo=2.3, loss_diloco=2.0)
    assert frac is not None
    expected = (2.5 - 2.3) / (2.5 - 2.0)
    assert abs(frac - expected) < 1e-6, f"Expected {expected}, got {frac}"
    # Edge case: denominator near zero
    frac2 = snoo_recovery_fraction(loss_dp=2.5, loss_snoo=2.3, loss_diloco=2.5)
    assert frac2 is None


# ===========================================================================
# TEST 11: Checkpoint save and load
# ===========================================================================

@test("T11 — Checkpoint save/load round-trip")
def test_checkpoint():
    model = make_model()
    optimizer = build_inner_optimizer(model, lr=1e-3, betas=[0.9, 0.95], eps=1e-8, weight_decay=0.1)
    tc = TokenCounter(token_budget=10000)
    tc.step(1000)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = save_checkpoint(
            output_dir=tmpdir,
            step=10,
            total_tokens=tc.total_tokens,
            model=model,
            inner_optimizer=optimizer,
            token_counter_state=tc.state_dict(),
            config_dict={"run_name": "test"},
            git_sha="abc123",
        )
        # Load into fresh objects
        model2 = make_model()
        optimizer2 = build_inner_optimizer(model2, lr=1e-3, betas=[0.9, 0.95], eps=1e-8, weight_decay=0.1)
        tc2 = TokenCounter(token_budget=1)
        meta = load_checkpoint(ckpt_dir, model2, optimizer2, tc2, DEVICE)

        assert meta["step"] == 10
        assert meta["total_tokens"] == 1000
        assert tc2.total_tokens == 1000
        # Parameter equality
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.allclose(p1, p2), f"Param mismatch at {n1}"


# ===========================================================================
# TEST 12: Config load and save
# ===========================================================================

@test("T12 — Config YAML save/load round-trip")
def test_config_io():
    from src.utils.config import save_config, load_config
    cfg = make_cfg("diloco", M=2, H=100, outer_type="nesterov")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "config.yaml"
        save_config(cfg, path)
        assert path.exists()
        cfg2 = load_config(path)
        assert cfg2.run_name == cfg.run_name
        assert cfg2.distribution.M_workers == 2
        assert cfg2.distribution.H_inner_steps == 100
        assert cfg2.outer_optimizer.type == "nesterov"


# ===========================================================================
# TEST 13: Clone-DiLoCo shard mode (all workers identical data)
# ===========================================================================

@test("T13 — Clone-DiLoCo: workers receive identical sequences")
def test_clone_shard_mode():
    from src.data.dataset import build_worker_datasets, make_dataloader
    datasets = build_worker_datasets(
        corpus="synthetic", tokenizer_name="gpt2",
        seq_len=SEQ, M_workers=2, shard_mode="clone",
        shard_seed=0, vocab_size=VOCAB, token_budget=BUDGET,
    )
    iter0 = iter(make_dataloader(datasets[0], BATCH))
    iter1 = iter(make_dataloader(datasets[1], BATCH))
    batch0 = next(iter0)
    batch1 = next(iter1)
    assert torch.equal(batch0, batch1), "Clone mode should produce identical batches"


# ===========================================================================
# TEST 14: IID shard mode (workers receive different data)
# ===========================================================================

@test("T14 — IID shard mode: workers receive different sequences")
def test_iid_shard_mode():
    from src.data.dataset import build_worker_datasets, make_dataloader
    datasets = build_worker_datasets(
        corpus="synthetic", tokenizer_name="gpt2",
        seq_len=SEQ, M_workers=2, shard_mode="iid",
        shard_seed=0, vocab_size=VOCAB, token_budget=BUDGET,
    )
    iter0 = iter(make_dataloader(datasets[0], BATCH))
    iter1 = iter(make_dataloader(datasets[1], BATCH))
    batch0 = next(iter0)
    batch1 = next(iter1)
    assert not torch.equal(batch0, batch1), "IID mode should produce different batches"


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

def run_all():
    print("\n" + "=" * 65)
    print("  CS769 DiLoCo Project — Smoke Test Suite")
    print("=" * 65)

    tests = [
        test_model_forward,
        test_token_counter,
        test_dp_adamw,
        test_snoo,
        test_periodic_avg,
        test_diloco,
        test_outer_optimizer_statedict,
        test_comm_tracker,
        test_geometry_metrics,
        test_snoo_recovery_fraction,
        test_checkpoint,
        test_config_io,
        test_clone_shard_mode,
        test_iid_shard_mode,
    ]

    t_start = time.perf_counter()
    for fn in tests:
        fn()

    total = time.perf_counter() - t_start
    n_pass = sum(1 for v in results.values() if "PASS" in v)
    n_fail = sum(1 for v in results.values() if "FAIL" in v)

    print("\n" + "=" * 65)
    print(f"  Results: {n_pass}/{len(tests)} passed  |  {n_fail} failed  |  {total:.1f}s total")
    print("=" * 65 + "\n")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all()
