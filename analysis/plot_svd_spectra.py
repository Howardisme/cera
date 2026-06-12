#!/usr/bin/env python3
"""
plot_svd_spectra.py -- Spectral Signature: singular value spectrum per rank.

Reads an SVD spectrum JSON (from analyze_svd.py) and plots the normalized
singular value spectra (log-scale y) for each (method, rank) combination,
showing how quickly each adapter's spectrum decays. A heavy tail indicates a
broad, distributed representational subspace; a sharp drop indicates rank
collapse.

Usage
-----
  python analysis/plot_svd_spectra.py \\
      --svd_json results/svd_spectra_orca.json \\
      --output results/svd_signature_orca.pdf

  python analysis/plot_svd_spectra.py \\
      --svd_json results/svd_spectra.json \\
      --ranks 16 64 128 512 \\
      --output results/svd_signature_math.pdf
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ── style ─────────────────────────────────────────────────────────────────────

METHOD_COLOR = {
    "CeRA": "#e05c2a",
    "LoRA": "#2a6ee0",
}

# Lighter -> darker with increasing rank so the rank ordering is readable.
RANK_ALPHA = {16: 0.35, 64: 0.55, 128: 0.75, 512: 1.0}


# ── data loading ──────────────────────────────────────────────────────────────

def _load_spectra(
    svd_json: str,
    methods: List[str],
    ranks: List[int],
) -> Dict[Tuple[str, int], np.ndarray]:
    """
    Returns {(method, rank): spectrum}, averaging over multiple runs of the
    same (method, rank). Spectra are L1-normalized by analyze_svd.py.
    """
    with open(svd_json) as f:
        entries = json.load(f)

    acc: Dict[Tuple[str, int], List[np.ndarray]] = {}
    for entry in entries:
        config = entry.get("config", {})
        method = config.get("type", config.get("model_type", ""))
        rank = int(config.get("rank", 0))
        spectrum = entry.get("spectrum")
        if method not in methods or not spectrum:
            continue
        if ranks and rank not in ranks:
            continue
        acc.setdefault((method, rank), []).append(np.array(spectrum, dtype=np.float64))

    data = {}
    for key, specs in acc.items():
        n = min(len(s) for s in specs)
        data[key] = np.mean([s[:n] for s in specs], axis=0)
    return data


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot singular value spectra (spectral signature) per adapter rank.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--svd_json", required=True,
        help="SVD spectrum JSON from analyze_svd.py.",
    )
    p.add_argument(
        "--methods", nargs="+", default=["CeRA", "LoRA"],
        help="Adapter methods to include.",
    )
    p.add_argument(
        "--ranks", nargs="+", type=int, default=[],
        help="Ranks to include (default: all found in the JSON).",
    )
    p.add_argument(
        "--xmax", type=int, default=None,
        help="Truncate the x axis at this singular value index.",
    )
    p.add_argument(
        "--rank_limit", type=int, default=None,
        help="Annotate this x position with an arrow labeled 'Rank Limit (N)'.",
    )
    p.add_argument(
        "--output", default="results/svd_signature.pdf",
        help="Output figure path (.pdf, .png, .svg).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not Path(args.svd_json).exists():
        print(f"[ERROR] SVD JSON not found: {args.svd_json}")
        sys.exit(1)

    data = _load_spectra(args.svd_json, args.methods, args.ranks)
    if not data:
        print("[ERROR] No matching spectra found in the JSON.")
        sys.exit(1)
    print(f"[INFO] Spectra loaded: {len(data)} (method, rank) combinations")

    fig, ax = plt.subplots(figsize=(6, 4))
    for (method, rank) in sorted(data.keys(), key=lambda k: (k[0], k[1])):
        spectrum = data[(method, rank)]
        # Zero singular values cannot be shown on a log axis
        spectrum = np.clip(spectrum, 1e-12, None)
        ax.plot(
            np.arange(1, len(spectrum) + 1),
            spectrum,
            color=METHOD_COLOR.get(method),
            alpha=RANK_ALPHA.get(rank, 1.0),
            linestyle="-" if method == "CeRA" else "--",
            linewidth=1.8,
            label=f"{method} R{rank}",
        )

    ax.set_xlabel("Singular Value Index", fontsize=12)
    ax.set_ylabel("Singular Value (log scale)", fontsize=12)
    ax.set_yscale("log")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    if args.xmax:
        ax.set_xlim(0, args.xmax)
    if args.rank_limit:
        ax.axvline(args.rank_limit, color="black", linestyle="--", linewidth=2.2)
        # Arrow at the rank-limit cliff; positions in log-y space.
        ymin, ymax = ax.get_ylim()
        log_mid = 0.5 * (np.log10(ymin) + np.log10(ymax))
        y_arrow = 10 ** log_mid
        y_text = 10 ** (log_mid + 0.35 * (np.log10(ymax) - log_mid))
        ax.annotate(
            f"Rank Limit ({args.rank_limit})",
            xy=(args.rank_limit, y_arrow),
            xytext=(args.rank_limit * 0.55, y_text),
            fontsize=10,
            fontweight="bold",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="black", linewidth=1.2),
        )

    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[SUCCESS] Figure saved -> {out}")


if __name__ == "__main__":
    main()
