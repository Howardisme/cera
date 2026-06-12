#!/usr/bin/env python3
"""
plot_training_curves.py -- Validation PPL curves during training.

Reads the training-log history of multiple runs and plots validation
perplexity against data seen, one panel per learning rate. Used for the
dropout robustness figure (paper Fig. 2): CeRA dropout grid vs LoRA under
suboptimal and optimal learning rates.

Usage
-----
  python analysis/plot_training_curves.py \\
      --results_dir results --dataset orca --rank 128 \\
      --lrs 1e-4 5e-4 \\
      --output results/training_curves_dropout.pdf
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


# ── style ─────────────────────────────────────────────────────────────────────

# CeRA: orange family, darker with higher dropout. LoRA: blue dashed.
CERA_DROPOUT_COLOR = {
    0.0: "#f2b27e",
    0.1: "#ea8c4e",
    0.2: "#e05c2a",
    0.3: "#a83c14",
}
LORA_COLOR = "#2a6ee0"


def _lr_decimal(lr: str) -> str:
    """'1e-4' -> '0.0001' (directory naming used by train.py)."""
    return str(float(lr))


# ── data loading ──────────────────────────────────────────────────────────────

def _load_runs(
    results_dir: str,
    dataset: str,
    rank: int,
    lr: str,
    methods: List[str],
) -> Dict[Tuple[str, float], List[Tuple[int, float]]]:
    """
    Collect validation PPL curves for all runs matching (dataset, rank, lr).
    Returns {(method, dropout): [(data_seen, test_ppl), ...]}.
    For duplicate (method, dropout) runs, the most recent directory wins.
    """
    lr_dec = _lr_decimal(lr)
    curves: Dict[Tuple[str, float], List[Tuple[int, float]]] = {}
    seen_dir: Dict[Tuple[str, float], str] = {}

    for exp_dir in sorted(Path(results_dir).iterdir()):
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        # startswith mirrors scan_experiment_dirs: quarantined runs renamed
        # with a prefix (e.g. ABORTED_Exp_...) must not be picked up.
        if not name.startswith("Exp_"):
            continue
        if f"_{dataset}_" not in name or f"_R{rank}_lr{lr_dec}_" not in name:
            continue

        for method in methods:
            log_file = exp_dir / method / f"{method}_log.json"
            if not log_file.exists():
                continue
            try:
                with open(log_file) as f:
                    log = json.load(f)
                config = log.get("config", {})
                dropout = float(config.get("dropout", 0.0))
                history = log.get("history", [])
                points = [
                    (h["data_seen"], h["new_task_target"]["test_ppl"])
                    for h in history
                    if h.get("new_task_target", {}).get("test_ppl") is not None
                ]
                if not points:
                    continue
                key = (method, dropout)
                # Directory names end in a timestamp; sorted() iteration means
                # a later directory for the same config overwrites an earlier one.
                curves[key] = points
                seen_dir[key] = name
            except Exception as exc:
                print(f"[WARN] Skipping {name}/{method}: {exc}")

    for key, d in sorted(seen_dir.items()):
        print(f"[INFO] {key[0]} D={key[1]} <- {d}")
    return curves


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot validation PPL curves during training, one panel per LR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results_dir", default="results",
                   help="Root results/ directory.")
    p.add_argument("--dataset", default="orca",
                   help="Dataset tag to filter runs (e.g. orca, math).")
    p.add_argument("--rank", type=int, default=128,
                   help="Adapter rank to filter runs.")
    p.add_argument("--lrs", nargs="+", default=["1e-4", "5e-4"],
                   help="Learning rates, one panel each (scientific notation).")
    p.add_argument("--methods", nargs="+", default=["CeRA", "LoRA"],
                   help="Adapter methods to include.")
    p.add_argument("--output", default="results/training_curves.pdf",
                   help="Output figure path (.pdf, .png, .svg).")
    return p.parse_args()


def main():
    args = parse_args()

    ncols = len(args.lrs)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]

    any_data = False
    for ax, lr in zip(axes, args.lrs):
        curves = _load_runs(args.results_dir, args.dataset, args.rank, lr, args.methods)
        if not curves:
            print(f"[WARN] No runs found for LR={lr}")
        for (method, dropout) in sorted(curves.keys()):
            xs, ys = zip(*curves[(method, dropout)])
            if method == "CeRA":
                color = CERA_DROPOUT_COLOR.get(dropout, "#e05c2a")
                label = f"CeRA D={dropout}"
                linestyle = "-"
            else:
                color = LORA_COLOR
                label = method
                linestyle = "--"
            ax.plot(xs, ys, color=color, linestyle=linestyle,
                    linewidth=1.8, marker="o", markersize=3, label=label)
            any_data = True

        ax.set_xlabel("Training Samples Seen", fontsize=12)
        # Log x: eval points are denser early in training. The data_seen=0
        # baseline point cannot appear on a log axis and is clipped.
        ax.set_xscale("log")
        ax.set_title(f"LR = {lr}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Validation Perplexity", fontsize=12)

    if not any_data:
        print("[ERROR] No matching runs found -- nothing to plot.")
        sys.exit(1)

    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[SUCCESS] Figure saved -> {out}")


if __name__ == "__main__":
    main()
