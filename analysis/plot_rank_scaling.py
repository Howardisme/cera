#!/usr/bin/env python3
"""
plot_rank_scaling.py -- Rank Scaling: PPL, Manifold Dimensionality, and
Effective Rank vs Adapter Rank.

Reads training logs (for PPL) and/or an SVD spectrum JSON (for the spectral
metrics) and plots how each metric scales with adapter rank for CeRA vs LoRA.

Metrics
-------
  ppl       Final test-set perplexity from training logs (requires --results_dir)
  manifold  Manifold dimensionality at --threshold variance (requires --svd_json)
  er        Effective Rank, Roy & Vetterli 2007 (requires --svd_json)
  both      Side-by-side subplots of ppl and manifold

Note on `er`: the spectrum stored by analyze_svd.py is averaged across the
tracked layers/projections, so the value here is the ER of the layer-averaged
spectrum, not the average of per-layer ERs. The two differ slightly; trends
across ranks are unaffected.

Usage
-----
  python analysis/plot_rank_scaling.py \\
      --results_dir results --dataset orca \\
      --svd_json results/svd_spectra_orca.json \\
      --metric both --output results/rank_scaling.pdf

  python analysis/plot_rank_scaling.py \\
      --svd_json results/svd_spectra_orca.json \\
      --metric er --output results/rank_scaling_er.pdf
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ── style ─────────────────────────────────────────────────────────────────────

METHOD_STYLE = {
    "CeRA": {"color": "#e05c2a", "marker": "o", "linestyle": "-",  "label": "CeRA"},
    "LoRA": {"color": "#2a6ee0", "marker": "s", "linestyle": "--", "label": "LoRA"},
}


# ── data loading ──────────────────────────────────────────────────────────────

def _load_ppl(
    results_dir: str,
    dataset: str,
    methods: List[str],
) -> Dict[str, Dict[int, float]]:
    """
    Scan results_dir for training logs that match dataset.
    For each (method, rank), keep the run with the lowest final test PPL.
    Returns {method: {rank: ppl}}.
    """
    data: Dict[str, Dict[int, float]] = {m: {} for m in methods}
    results_path = Path(results_dir)

    for exp_dir in sorted(results_path.iterdir()):
        if not exp_dir.is_dir():
            continue
        # startswith mirrors scan_experiment_dirs: quarantined runs renamed
        # with a prefix (e.g. ABORTED_Exp_...) must not be picked up.
        if not exp_dir.name.startswith("Exp_"):
            continue
        if f"_{dataset}_" not in exp_dir.name:
            continue

        for method in methods:
            log_file = exp_dir / method / f"{method}_log.json"
            if not log_file.exists():
                continue
            try:
                with open(log_file) as f:
                    log = json.load(f)
                config = log.get("config", {})
                rank = int(config.get("rank", 0))
                history = log.get("history", [])
                if rank == 0 or not history:
                    continue
                ppl = history[-1].get("new_task_target", {}).get("test_ppl")
                if ppl is None or ppl != ppl or ppl == float("inf"):
                    continue
                if rank not in data[method] or ppl < data[method][rank]:
                    data[method][rank] = float(ppl)
            except Exception:
                continue

    return data


def _compute_manifold_dim(spectrum: List[float], threshold: float) -> int:
    """
    Minimum number of singular values needed to explain `threshold` fraction of
    total variance. Input spectrum is L1-normalized (from get_singular_values).
    """
    s = np.array(spectrum, dtype=np.float64)
    var = s ** 2
    total = var.sum()
    if total < 1e-12:
        return 0
    cumvar = np.cumsum(var)
    k = int(np.sum(cumvar / total < threshold)) + 1
    return min(k, len(s))


def _compute_er(spectrum: List[float]) -> float:
    """
    Effective Rank (Roy & Vetterli 2007) of a singular value spectrum:
        ER = exp( -sum_i p_i log p_i ),  p_i = sigma_i / sum_j sigma_j
    Matches cera.metrics.compute_effective_rank, but operates on a stored
    spectrum instead of an activation matrix.
    """
    s = np.array(spectrum, dtype=np.float64)
    total = s.sum()
    if total < 1e-12:
        return 0.0
    p = s / total
    entropy = -np.sum(p * np.log(p + 1e-10))
    return float(np.exp(entropy))


def _load_spectrum_metric(
    svd_json: str,
    methods: List[str],
    metric_fn,
) -> Dict[str, Dict[int, float]]:
    """
    Read SVD spectrum JSON (from analyze_svd.py) and apply metric_fn to each
    spectrum, averaging over multiple runs of the same (method, rank).
    Returns {method: {rank: value}}.
    """
    data: Dict[str, Dict[int, float]] = {m: {} for m in methods}
    with open(svd_json) as f:
        entries = json.load(f)

    counts: Dict[str, Dict[int, int]] = {m: {} for m in methods}

    for entry in entries:
        config = entry.get("config", {})
        method = config.get("type", config.get("model_type", ""))
        if method not in methods:
            continue
        rank = int(config.get("rank", 0))
        spectrum = entry.get("spectrum")
        if not spectrum or rank == 0:
            continue
        val = float(metric_fn(spectrum))
        if rank in data[method]:
            data[method][rank] += val
            counts[method][rank] += 1
        else:
            data[method][rank] = val
            counts[method][rank] = 1

    # average over multiple runs for same rank
    for method in methods:
        for rank in data[method]:
            data[method][rank] /= counts[method][rank]

    return data


# ── plotting ──────────────────────────────────────────────────────────────────

def _plot_metric(
    ax,
    data: Dict[str, Dict[int, float]],
    methods: List[str],
    ylabel: str,
    title: str,
):
    for method in methods:
        pts = sorted(data[method].items())
        if not pts:
            print(f"[WARN] No data for method={method}")
            continue
        ranks, vals = zip(*pts)
        style = METHOD_STYLE.get(method, {})
        ax.plot(
            ranks, vals,
            marker=style.get("marker", "o"),
            linestyle=style.get("linestyle", "-"),
            color=style.get("color"),
            label=style.get("label", method),
            linewidth=2,
            markersize=6,
        )
    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=12)
    # Log spacing, but label ticks with the literal rank values (16, 64, ...)
    # instead of base-2 exponents.
    ax.set_xscale("log", base=2)
    all_ranks = sorted({r for m in methods for r in data.get(m, {})})
    if all_ranks:
        ax.set_xticks(all_ranks)
        ax.set_xticklabels([str(r) for r in all_ranks])
        ax.minorticks_off()
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot PPL and/or manifold dimensionality vs adapter rank.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--metric", choices=["ppl", "manifold", "er", "both"], default="both",
        help="Which metric(s) to plot. 'both' = ppl + manifold side by side.",
    )
    p.add_argument(
        "--results_dir", default="results",
        help="Root results/ directory (for --metric ppl or both).",
    )
    p.add_argument(
        "--dataset", default="orca",
        help="Dataset tag to filter training logs (e.g. orca, math).",
    )
    p.add_argument(
        "--svd_json", default=None,
        help="SVD spectrum JSON from analyze_svd.py (for --metric manifold, er, or both).",
    )
    p.add_argument(
        "--threshold", type=float, default=0.90,
        help="Variance threshold for manifold dimensionality.",
    )
    p.add_argument(
        "--methods", nargs="+", default=["CeRA", "LoRA"],
        help="Adapter methods to include.",
    )
    p.add_argument(
        "--output", default="results/rank_scaling.pdf",
        help="Output figure path (.pdf, .png, .svg).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    need_ppl      = args.metric in ("ppl", "both")
    need_manifold = args.metric in ("manifold", "both")
    need_er       = args.metric == "er"

    if need_manifold or need_er:
        if not args.svd_json:
            print(f"[ERROR] --svd_json required for {args.metric} metric.")
            sys.exit(1)
        if not Path(args.svd_json).exists():
            print(f"[ERROR] SVD JSON not found: {args.svd_json}")
            sys.exit(1)

    ppl_data      = _load_ppl(args.results_dir, args.dataset, args.methods) if need_ppl else {}
    manifold_data = (
        _load_spectrum_metric(
            args.svd_json, args.methods,
            lambda s: _compute_manifold_dim(s, args.threshold),
        )
        if need_manifold else {}
    )
    er_data = _load_spectrum_metric(args.svd_json, args.methods, _compute_er) if need_er else {}

    if need_ppl:
        total = sum(len(v) for v in ppl_data.values())
        print(f"[INFO] PPL data points loaded: {total}")
    if need_manifold:
        total = sum(len(v) for v in manifold_data.values())
        print(f"[INFO] Manifold data points loaded: {total}")
    if need_er:
        total = sum(len(v) for v in er_data.values())
        print(f"[INFO] ER data points loaded: {total}")

    ncols = 2 if args.metric == "both" else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    if need_ppl:
        _plot_metric(
            axes[0], ppl_data, args.methods,
            ylabel="Perplexity",
            title=f"Capacity Scaling ({args.dataset})",
        )
    if need_manifold:
        pct = int(args.threshold * 100)
        _plot_metric(
            axes[-1], manifold_data, args.methods,
            ylabel=f"Manifold Dimensionality ({pct}% Variance)",
            title="Spectral Dimensionality vs Rank",
        )
    if need_er:
        _plot_metric(
            axes[0], er_data, args.methods,
            ylabel="Effective Rank",
            title="",
        )

    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[SUCCESS] Figure saved -> {out}")


if __name__ == "__main__":
    main()
