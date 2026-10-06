"""
Attribution bar plot (Figure 3).

Final validation loss delta vs DP-AdamW for all methods:
  SW-AdamW, SNOO, PeriodicAvg, DiLoCo, Clone-DiLoCo

Usage::
    python analysis/plot_attribution.py --runs_dir runs/e2 --output figures/fig3_attribution.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


METHOD_LABELS = {
    "dp_adamw":    "DP-AdamW",
    "sw_adamw":    "SW-AdamW",
    "snoo":        "SNOO",
    "periodic_avg": "PeriodicAvg",
    "diloco":      "DiLoCo",
    "clone_diloco": "Clone-DiLoCo",
}

METHOD_COLORS = {
    "dp_adamw":     "#6c757d",
    "sw_adamw":     "#0077b6",
    "snoo":         "#00b4d8",
    "periodic_avg": "#e9c46a",
    "diloco":       "#e63946",
    "clone_diloco": "#f4a261",
}


def load_run_summaries(runs_dir: Path) -> dict[str, list[float]]:
    """Load final val loss per seed for each method from run_summary.csv/jsonl files."""
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
        # Try config.yaml in the same directory
        cfg_file = summary_file.parent / "config.yaml"
        if cfg_file.exists():
            try:
                with open(cfg_file, encoding="utf-8") as cf:
                    cfg_dict = yaml.safe_load(cf)
                    method = cfg_dict.get("distribution", {}).get("method")
            except Exception:
                pass

        if not method:
            # Try last row or run name
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

        # Find final validation loss
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


def plot_attribution(
    method_losses: dict[str, list[float]],
    output: Path,
    model_scale: str = "160M",
) -> None:
    # Compute stats
    dp_losses = method_losses.get("dp_adamw", [])
    dp_mean   = np.nanmean(dp_losses) if dp_losses else 0.0

    methods_ordered = ["sw_adamw", "snoo", "periodic_avg", "diloco", "clone_diloco"]
    labels, deltas, errors = [], [], []

    for m in methods_ordered:
        losses = method_losses.get(m)
        if not losses:
            continue
        arr = np.array(losses)
        mean = np.nanmean(arr)
        std  = np.nanstd(arr) if len(arr) > 1 else 0.0
        labels.append(METHOD_LABELS.get(m, m))
        deltas.append(dp_mean - mean if dp_losses else mean)
        errors.append(std)

    if not deltas:
        print("No valid method losses to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    colors = [METHOD_COLORS.get(m, "#888") for m in methods_ordered if m in method_losses]
    bars = ax.barh(labels, deltas, xerr=errors, color=colors,
                   edgecolor="white", linewidth=0.5, height=0.55, capsize=4)

    ax.axvline(0, color="#6c757d", linewidth=1.2, linestyle="--", alpha=0.8)
    ax.set_xlabel("Δ Validation Loss vs DP-AdamW (↓ negative = worse)" if dp_losses else "Validation Loss", color="white", fontsize=11)
    ax.set_title(
        f"Fig 3: Causal Decomposition — {model_scale} model\n"
        "Token/FLOP matched  |  H=100  |  M=4",
        color="white", fontsize=12, pad=12,
    )
    ax.tick_params(colors="white", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    min_val = min(deltas)
    max_val = max(deltas)
    padding = max((max_val - min_val) * 0.1, 0.05)
    ax.set_xlim(min_val - padding, max_val + padding)

    # DP baseline annotation
    if dp_losses:
        ax.text(
            0.98, 0.02, f"DP-AdamW val_loss = {dp_mean:.4f}",
            transform=ax.transAxes, ha="right", color="#6c757d", fontsize=9,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved -> {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="figures/fig3_attribution.png")
    parser.add_argument("--scale", type=str, default="160M")
    args = parser.parse_args()

    method_losses = load_run_summaries(Path(args.runs_dir))
    if not method_losses:
        print("No run summaries found. Run experiments first.")
        return
    plot_attribution(method_losses, Path(args.output), model_scale=args.scale)


if __name__ == "__main__":
    main()
