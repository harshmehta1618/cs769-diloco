"""
Compute SNOO recovery fraction (Experiment E2).

SNOO recovery fraction = (Loss_DP − Loss_SNOO) / (Loss_DP − Loss_DiLoCo)

Quantifies: how much of DiLoCo's quality gain is explained by the outer
Nesterov optimizer alone (SNOO), vs the worker diversity interaction.

Usage::
    python analysis/compute_snoo_frac.py --runs_dir runs/e2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import io

# Force UTF-8 output on Windows
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.metrics.evaluator import snoo_recovery_fraction


def collect_final_losses(runs_dir: Path) -> dict[str, list[float]]:
    import csv
    import yaml
    method_losses: dict[str, list[float]] = {}
    for summary_file in sorted(runs_dir.glob("**/run_summary.*")):
        if summary_file.suffix == ".jsonl":
            with open(summary_file, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        elif summary_file.suffix == ".csv":
            with open(summary_file, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        else:
            continue

        if not rows:
            continue

        method = None
        cfg_file = summary_file.parent / "config.yaml"
        if cfg_file.exists():
            try:
                with open(cfg_file, encoding="utf-8") as cf:
                    cfg_dict = yaml.safe_load(cf)
                    method = cfg_dict.get("distribution", {}).get("method")
            except Exception:
                pass

        if not method:
            for r in reversed(rows):
                if "method" in r and r["method"]:
                    method = r["method"]
                    break
                run_name = r.get("run", "")
                for m_candidate in ["clone_diloco", "periodic_avg", "dp_adamw", "sw_adamw", "diloco", "snoo"]:
                    if m_candidate in run_name or m_candidate in summary_file.parent.name:
                        method = m_candidate
                        break
                if method:
                    break

        if not method:
            method = "unknown"

        loss_val = None
        for r in reversed(rows):
            val = r.get("final_val_loss") or r.get("val_loss")
            if val is not None and str(val).strip() != "":
                loss_val = val
                break

        if loss_val is not None:
            try:
                method_losses.setdefault(method, []).append(float(loss_val))
            except ValueError:
                pass

    return method_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", required=True)
    args = parser.parse_args()

    losses = collect_final_losses(Path(args.runs_dir))
    print("\n" + "=" * 55)
    print("  SNOO Recovery Fraction Analysis")
    print("=" * 55)

    required = {"dp_adamw", "snoo", "diloco"}
    missing  = required - set(losses.keys())
    if missing:
        print(f"  Missing methods: {missing}")
        print("  Run E2 experiments first.")
        return

    dp_mean     = np.mean(losses["dp_adamw"])
    snoo_mean   = np.mean(losses["snoo"])
    diloco_mean = np.mean(losses["diloco"])

    frac = snoo_recovery_fraction(dp_mean, snoo_mean, diloco_mean)

    print(f"  DP-AdamW  val_loss = {dp_mean:.4f} +/- {np.std(losses['dp_adamw']):.4f}")
    print(f"  SNOO      val_loss = {snoo_mean:.4f} +/- {np.std(losses['snoo']):.4f}")
    print(f"  DiLoCo    val_loss = {diloco_mean:.4f} +/- {np.std(losses['diloco']):.4f}")
    print()

    if frac is None:
        print("  Recovery fraction: UNDEFINED (DiLoCo not better than DP)")
        print("  Interpretation: no quality free lunch to attribute.")
    else:
        print(f"  SNOO recovery fraction: {frac:.3f}  ({frac*100:.1f}%)")
        if frac > 0.8:
            interp = "Most gain is from outer Nesterov (SNOO ~= DiLoCo > DP)"
        elif frac > 0.4:
            interp = "Outer optimizer + worker diversity both contribute"
        else:
            interp = "Worker diversity dominates; Nesterov adds little"
        print(f"  Interpretation: {interp}")

    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
