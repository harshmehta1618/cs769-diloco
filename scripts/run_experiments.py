"""
Experiment Runner Script.

Automates running experiment matrices across seeds, methods, and configurations.

Usage::
    # Run E0 Harness validation
    python scripts/run_experiments.py --experiment E0

    # Run E1 Pilot (60M)
    python scripts/run_experiments.py --experiment E1

    # Run E2 Main Causal Decomposition (160M, 5 methods x 3 seeds)
    python scripts/run_experiments.py --experiment E2 --seeds 0 1 2

    # Run E3 Horizon sensitivity
    python scripts/run_experiments.py --experiment E3 --horizons 50 100 200 500

    # Run E4 Worker scaling
    python scripts/run_experiments.py --experiment E4 --workers 1 2 4 8

    # Run E5 Clone-DiLoCo ablation
    python scripts/run_experiments.py --experiment E5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> bool:
    print(f"\n[RUNNING] {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[FAILED] Command exited with code {res.returncode}")
        return False
    return True


def run_e0():
    print("=== Running E0: Harness Equivalence ===")
    return run_cmd([sys.executable, "scripts/train.py", "--config", "configs/e0_harness.yaml"])


def run_e1(seeds: list[int]):
    print("=== Running E1: 60M Pilot ===")
    methods = ["dp_adamw", "snoo", "diloco"]
    success = True
    for method in methods:
        for seed in seeds:
            run_name = f"e1_pilot_{method}_seed{seed}"
            cmd = [
                sys.executable, "scripts/train.py",
                "--config", "configs/e1_pilot_60m.yaml",
                "--override",
                f"run_name={run_name}",
                f"distribution.method={method}",
                f"runtime.seed={seed}",
            ]
            if not run_cmd(cmd):
                success = False
    return success


def run_e2(seeds: list[int]):
    print("=== Running E2: Main Causal Decomposition (160M) ===")
    methods = ["dp_adamw", "sw_adamw", "snoo", "periodic_avg", "diloco"]
    success = True
    for method in methods:
        for seed in seeds:
            run_name = f"e2_{method}_160m_seed{seed}"
            cmd = [
                sys.executable, "scripts/train.py",
                "--config", "configs/e2_attribution_160m.yaml",
                "--override",
                f"run_name={run_name}",
                f"distribution.method={method}",
                f"runtime.seed={seed}",
            ]
            if not run_cmd(cmd):
                success = False
    return success


def run_e3(horizons: list[int], seeds: list[int]):
    print("=== Running E3: Horizon H Sensitivity ===")
    methods = ["diloco", "periodic_avg"]
    success = True
    for method in methods:
        for h in horizons:
            for seed in seeds:
                run_name = f"e3_{method}_h{h}_seed{seed}"
                cmd = [
                    sys.executable, "scripts/train.py",
                    "--config", "configs/e3_horizon_h.yaml",
                    "--override",
                    f"run_name={run_name}",
                    f"distribution.method={method}",
                    f"distribution.H_inner_steps={h}",
                    f"runtime.seed={seed}",
                ]
                if not run_cmd(cmd):
                    success = False
    return success


def run_e4(workers: list[int], seeds: list[int]):
    print("=== Running E4: Worker M Scaling ===")
    methods = ["diloco", "periodic_avg"]
    success = True
    for method in methods:
        for m in workers:
            for seed in seeds:
                run_name = f"e4_{method}_m{m}_seed{seed}"
                cmd = [
                    sys.executable, "scripts/train.py",
                    "--config", "configs/e4_worker_scaling.yaml",
                    "--override",
                    f"run_name={run_name}",
                    f"distribution.method={method}",
                    f"distribution.M_workers={m}",
                    f"runtime.seed={seed}",
                ]
                if not run_cmd(cmd):
                    success = False
    return success


def run_e5(seeds: list[int]):
    print("=== Running E5: Clone-DiLoCo Data Diversity Ablation ===")
    success = True
    for seed in seeds:
        run_name = f"e5_clone_diloco_seed{seed}"
        cmd = [
            sys.executable, "scripts/train.py",
            "--config", "configs/e5_clone_diloco.yaml",
            "--override",
            f"run_name={run_name}",
            f"runtime.seed={seed}",
        ]
        if not run_cmd(cmd):
            success = False
    return success


def main():
    parser = argparse.ArgumentParser(description="CS769 Experiment Suite Runner")
    parser.add_argument("--experiment", choices=["E0", "E1", "E2", "E3", "E4", "E5", "all"], default="E0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="Random seeds to evaluate")
    parser.add_argument("--horizons", nargs="+", type=int, default=[50, 100, 200, 500], help="H values for E3")
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4, 8], help="M worker counts for E4")
    args = parser.parse_args()

    if args.experiment in ("E0", "all"):
        run_e0()
    if args.experiment in ("E1", "all"):
        run_e1(args.seeds)
    if args.experiment in ("E2", "all"):
        run_e2(args.seeds)
    if args.experiment in ("E3", "all"):
        run_e3(args.horizons, args.seeds)
    if args.experiment in ("E4", "all"):
        run_e4(args.workers, args.seeds)
    if args.experiment in ("E5", "all"):
        run_e5(args.seeds)


if __name__ == "__main__":
    main()
