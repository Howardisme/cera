#!/usr/bin/env python3
"""
Diagnostic for MATH-500 pass@10 anomaly.

CeRA leads pass@1 on MATH-500 (15.8% vs LoRA 11.8%) but does NOT lead pass@10
(44.0% vs DoRA 44.6%). This script uses the already-saved
math500_pass10.jsonl for each cell to test three hypotheses without any
additional compute:

  H1  Output collapse:   CeRA's 10 samples per problem are near-identical
                         (low distinct-answer count) -- greedy accurate but
                         no representational expansion under sampling.
  H2  Mode collapse on wrong direction:   When wrong, CeRA converges on the
                         same wrong answer across all 10 samples.
  H3  Exploration failure:   Samples are diverse but none find the right
                         answer -- model is not in a solution regime.

Reads:  results/eval_outputs/<cell>/math500_pass10.jsonl  for each cell in CELLS

Run from repo root:
    python analysis/analyze_pass10_diversity.py
"""
import json
import statistics
from pathlib import Path


CELLS = [
    ("LoRA", "r64_lora_a64_lr3e-4"),
    ("DoRA", "r64_dora_a64_lr3e-4"),
    ("CeRA", "r64_cera_lr3e-4"),
]

BASE = Path("results/eval_outputs")


def load_records(cell_dir):
    path = BASE / cell_dir / "math500_pass10.jsonl"
    with open(path) as fh:
        return [json.loads(line) for line in fh]


def analyze_problem(rec):
    """Per-problem diversity stats over the 10 samples."""
    samples = rec["samples"]
    # Normalize None -> a single sentinel so all extraction failures count as
    # one "no answer" bucket rather than 10 different Nones.
    answers = ["__NO_ANSWER__" if s.get("extracted_answer") is None
               else s["extracted_answer"] for s in samples]
    distinct = set(answers)
    gen_lens = [len(s.get("generated", "")) for s in samples]
    none_count = sum(1 for s in samples if s.get("extracted_answer") is None)
    n_correct = rec.get("n_correct", 0)
    return {
        "distinct_count":    len(distinct),
        "all_same":          len(distinct) == 1,
        "all_extract_fail":  none_count == len(samples),
        "extraction_failures_out_of_10": none_count,
        "mean_gen_len":      statistics.mean(gen_lens),
        "n_correct":         n_correct,
        "any_correct":       n_correct > 0,
    }


def summarize(per_problem):
    n = len(per_problem)
    wrong  = [p for p in per_problem if p["n_correct"] == 0]
    correct_any = [p for p in per_problem if p["n_correct"] > 0]
    return {
        "n_problems":              n,
        "mean_distinct":           statistics.mean(p["distinct_count"] for p in per_problem),
        "median_distinct":         statistics.median(p["distinct_count"] for p in per_problem),
        "pct_all_same":            100 * sum(p["all_same"] for p in per_problem) / n,
        "pct_all_extract_fail":    100 * sum(p["all_extract_fail"] for p in per_problem) / n,
        "mean_extract_fails_per10": statistics.mean(p["extraction_failures_out_of_10"] for p in per_problem),
        "mean_gen_len":            statistics.mean(p["mean_gen_len"] for p in per_problem),
        "any_correct_rate":        100 * sum(p["any_correct"] for p in per_problem) / n,
        # Failure-mode diagnostics (only over problems where n_correct == 0)
        "n_wrong":                 len(wrong),
        "mean_distinct_when_wrong": (statistics.mean(p["distinct_count"] for p in wrong)
                                     if wrong else float("nan")),
        "pct_mode_collapse_wrong": (100 * sum(p["distinct_count"] == 1 for p in wrong) / len(wrong)
                                    if wrong else float("nan")),
        # Success-mode diagnostics (only over problems where at least one of 10 was correct)
        "n_any_correct":           len(correct_any),
        "mean_distinct_when_any_correct": (statistics.mean(p["distinct_count"] for p in correct_any)
                                           if correct_any else float("nan")),
    }


def main():
    all_results = []
    for name, cell_dir in CELLS:
        records = load_records(cell_dir)
        per_prob = [analyze_problem(r) for r in records]
        summary = summarize(per_prob)
        all_results.append((name, summary))

    # -------- Overall diversity table --------
    print("=" * 82)
    print("MATH-500 pass@10 sample-diversity summary (N=500 problems, 10 samples/problem)")
    print("=" * 82)
    print(f"{'Method':<6} {'meanDist':>9} {'medDist':>8} {'%allSame':>9} "
          f"{'%allExtFail':>12} {'meanExtFail/10':>15} {'meanGenLen':>11} {'anyCorr%':>9}")
    print("-" * 82)
    for name, s in all_results:
        print(f"{name:<6} "
              f"{s['mean_distinct']:>9.2f} "
              f"{s['median_distinct']:>8.1f} "
              f"{s['pct_all_same']:>8.1f}% "
              f"{s['pct_all_extract_fail']:>11.1f}% "
              f"{s['mean_extract_fails_per10']:>15.2f} "
              f"{s['mean_gen_len']:>11.0f} "
              f"{s['any_correct_rate']:>8.1f}%")

    # -------- Failure-mode breakdown --------
    print()
    print("Failure mode (problems where all 10 samples wrong, n_correct == 0):")
    print(f"{'Method':<6} {'N_wrong':>8} {'meanDist(wrong)':>16} {'%modeCollapse(wrong)':>22}")
    print("-" * 60)
    for name, s in all_results:
        print(f"{name:<6} {s['n_wrong']:>8d} "
              f"{s['mean_distinct_when_wrong']:>16.2f} "
              f"{s['pct_mode_collapse_wrong']:>21.1f}%")

    # -------- Success-mode breakdown --------
    print()
    print("Success mode (problems where at least 1 of 10 was correct):")
    print(f"{'Method':<6} {'N_any_correct':>14} {'meanDist(anyCorr)':>18}")
    print("-" * 50)
    for name, s in all_results:
        print(f"{name:<6} {s['n_any_correct']:>14d} "
              f"{s['mean_distinct_when_any_correct']:>18.2f}")

    # -------- Interpretation cheatsheet --------
    print()
    print("=" * 82)
    print("Interpretation cheatsheet")
    print("=" * 82)
    print("H1 (output collapse):   CeRA meanDist << LoRA/DoRA meanDist")
    print("H2 (wrong mode collapse): CeRA %modeCollapse(wrong) > LoRA/DoRA")
    print("H3 (exploration fail):  CeRA meanDist(wrong) similar to others but any_correct% lower")
    print("If CeRA mean_gen_len is much shorter: samples truncate early, less reasoning space.")


if __name__ == "__main__":
    main()
