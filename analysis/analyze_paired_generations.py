#!/usr/bin/env python3
"""Paired CeRA-vs-LoRA analysis over existing evaluation JSONL files.

The script does not run a model. It reuses saved generations to report paired
accuracy, exact McNemar tests, question-bootstrap confidence intervals,
cross-seed stability, simple generation diagnostics, and reviewable cases.

Example:
  python3 analysis/analyze_paired_generations.py \
    --pair 42 path/to/cera_seed42.jsonl path/to/lora_seed42.jsonl \
    --pair 43 path/to/cera_seed43.jsonl path/to/lora_seed43.jsonl \
    --output-dir results/paired_math_hard_qv
"""

import argparse
import json
import random
import statistics
from collections import Counter
from math import comb
from pathlib import Path

from regrade_community import _HAVE_MV, extract, grade


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze paired CeRA/LoRA generations across training seeds."
    )
    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        required=True,
        metavar=("SEED", "CERA_JSONL", "LORA_JSONL"),
        help="Training seed followed by matching CeRA and LoRA JSONL files.",
    )
    parser.add_argument(
        "--grading",
        choices=("community", "stored"),
        default="community",
        help="Re-grade saved text or trust the correctness stored by evaluate.py.",
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260912)
    parser.add_argument("--max-review-cases", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "prompt" not in record or "samples" not in record:
                raise ValueError(f"{path}:{line_number}: missing prompt or samples")
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def correctness(record, grading):
    if grading == "stored":
        return int(record.get("n_correct", 0) > 0)
    return int(
        any(
            grade(extract(sample.get("generated", "")), record.get("gold_answer"))
            for sample in record.get("samples", [])
        )
    )


def first_generation(record):
    samples = record.get("samples") or []
    return samples[0].get("generated", "") if samples else ""


def align_pair(cera_records, lora_records, seed):
    cera_by_prompt = {record["prompt"]: record for record in cera_records}
    lora_by_prompt = {record["prompt"]: record for record in lora_records}
    if len(cera_by_prompt) != len(cera_records):
        raise ValueError(f"Seed {seed}: duplicate CeRA prompts")
    if len(lora_by_prompt) != len(lora_records):
        raise ValueError(f"Seed {seed}: duplicate LoRA prompts")
    if set(cera_by_prompt) != set(lora_by_prompt):
        only_cera = len(set(cera_by_prompt) - set(lora_by_prompt))
        only_lora = len(set(lora_by_prompt) - set(cera_by_prompt))
        raise ValueError(
            f"Seed {seed}: prompt mismatch (CeRA-only={only_cera}, LoRA-only={only_lora})"
        )

    aligned = []
    for prompt, cera in cera_by_prompt.items():
        lora = lora_by_prompt[prompt]
        if str(cera.get("gold_answer")) != str(lora.get("gold_answer")):
            raise ValueError(f"Seed {seed}: gold mismatch for prompt {prompt[:80]!r}")
        aligned.append((prompt, cera, lora))
    return aligned


def exact_mcnemar(cera_only, lora_only):
    discordant = cera_only + lora_only
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, index) for index in range(min(cera_only, lora_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def percentile(values, probability):
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_delta(outcomes, iterations, rng):
    if iterations <= 0:
        return None
    differences = [cera - lora for cera, lora in outcomes]
    count = len(differences)
    estimates = [
        100.0 * sum(rng.choice(differences) for _ in range(count)) / count
        for _ in range(iterations)
    ]
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def method_diagnostics(records, outcomes):
    generated = [first_generation(record) for record in records]
    extracted = [extract(text) for text in generated]
    return {
        "accuracy": sum(outcomes) / len(outcomes),
        "n_correct": sum(outcomes),
        "extraction_failure_rate": sum(value is None for value in extracted) / len(extracted),
        "mean_generated_characters": statistics.fmean(map(len, generated)),
        "median_generated_characters": statistics.median(map(len, generated)),
        "mean_answer_markers": statistics.fmean(text.count("####") for text in generated),
    }


def analyze_seed(seed, cera_path, lora_path, grading, iterations, rng):
    cera_records = load_jsonl(cera_path)
    lora_records = load_jsonl(lora_path)
    aligned = align_pair(cera_records, lora_records, seed)

    rows = []
    for prompt, cera, lora in aligned:
        cera_correct = correctness(cera, grading)
        lora_correct = correctness(lora, grading)
        rows.append(
            {
                "key": prompt,
                "id": cera.get("id"),
                "prompt": prompt,
                "gold_answer": cera.get("gold_answer"),
                "cera_correct": cera_correct,
                "lora_correct": lora_correct,
                "cera_generated": first_generation(cera),
                "lora_generated": first_generation(lora),
                "cera_extracted": extract(first_generation(cera)),
                "lora_extracted": extract(first_generation(lora)),
            }
        )

    outcomes = [(row["cera_correct"], row["lora_correct"]) for row in rows]
    cera_only = sum(cera == 1 and lora == 0 for cera, lora in outcomes)
    lora_only = sum(cera == 0 and lora == 1 for cera, lora in outcomes)
    cera_outcomes = [outcome[0] for outcome in outcomes]
    lora_outcomes = [outcome[1] for outcome in outcomes]
    sample_size = len(rows)

    summary = {
        "seed": seed,
        "n": sample_size,
        "cera": method_diagnostics([row[1] for row in aligned], cera_outcomes),
        "lora": method_diagnostics([row[2] for row in aligned], lora_outcomes),
        "delta_percentage_points": 100.0 * (sum(cera_outcomes) - sum(lora_outcomes)) / sample_size,
        "bootstrap_95_ci_percentage_points": bootstrap_delta(outcomes, iterations, rng),
        "cera_only_correct": cera_only,
        "lora_only_correct": lora_only,
        "mcnemar_exact_p": exact_mcnemar(cera_only, lora_only),
    }
    return summary, rows


def jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def cross_seed_summary(rows_by_seed):
    common_keys = set.intersection(
        *(set(row["key"] for row in rows) for rows in rows_by_seed.values())
    )
    by_seed = {
        seed: {row["key"]: row for row in rows}
        for seed, rows in rows_by_seed.items()
    }
    seeds = list(rows_by_seed)
    cera_only_sets = {
        seed: {
            key for key in common_keys
            if by_seed[seed][key]["cera_correct"] and not by_seed[seed][key]["lora_correct"]
        }
        for seed in seeds
    }
    lora_only_sets = {
        seed: {
            key for key in common_keys
            if by_seed[seed][key]["lora_correct"] and not by_seed[seed][key]["cera_correct"]
        }
        for seed in seeds
    }

    pairwise = []
    for left_index, left_seed in enumerate(seeds):
        for right_seed in seeds[left_index + 1:]:
            pairwise.append(
                {
                    "seeds": [left_seed, right_seed],
                    "cera_only_jaccard": jaccard(cera_only_sets[left_seed], cera_only_sets[right_seed]),
                    "lora_only_jaccard": jaccard(lora_only_sets[left_seed], lora_only_sets[right_seed]),
                }
            )

    stable_cera = set.intersection(*(cera_only_sets[seed] for seed in seeds))
    stable_lora = set.intersection(*(lora_only_sets[seed] for seed in seeds))
    agreement = {}
    for method in ("cera", "lora"):
        field = f"{method}_correct"
        agreement[method] = sum(
            len({by_seed[seed][key][field] for seed in seeds}) == 1
            for key in common_keys
        ) / len(common_keys)

    return {
        "common_questions": len(common_keys),
        "mean_delta_percentage_points": statistics.fmean(
            100.0 * sum(
                by_seed[seed][key]["cera_correct"] - by_seed[seed][key]["lora_correct"]
                for key in common_keys
            ) / len(common_keys)
            for seed in seeds
        ),
        "stable_cera_only_correct": len(stable_cera),
        "stable_lora_only_correct": len(stable_lora),
        "correctness_agreement_across_seeds": agreement,
        "pairwise_directional_jaccard": pairwise,
    }, stable_cera, stable_lora


def write_disagreements(path, rows_by_seed, stable_cera, stable_lora):
    with path.open("w", encoding="utf-8") as handle:
        for seed, rows in rows_by_seed.items():
            for row in rows:
                if row["cera_correct"] == row["lora_correct"]:
                    continue
                output = dict(row)
                output["seed"] = seed
                output["stable_direction_across_seeds"] = (
                    "cera" if row["key"] in stable_cera
                    else "lora" if row["key"] in stable_lora
                    else None
                )
                output.pop("key")
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")


def write_review(path, rows_by_seed, stable_cera, stable_lora, limit):
    candidates = []
    for seed, rows in rows_by_seed.items():
        for row in rows:
            if row["cera_correct"] == row["lora_correct"]:
                continue
            stable = row["key"] in stable_cera or row["key"] in stable_lora
            candidates.append((not stable, str(seed), row))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]["id"] or -1))

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# CeRA vs LoRA disagreement review\n\n")
        for _, seed, row in candidates[:limit]:
            if row["key"] in stable_cera:
                stability = "stable CeRA-only across all seeds"
            elif row["key"] in stable_lora:
                stability = "stable LoRA-only across all seeds"
            else:
                stability = "seed-specific disagreement"
            handle.write(f"## Seed {seed}, ID {row['id']} ({stability})\n\n")
            handle.write(f"**Gold:** {row['gold_answer']}\n\n")
            handle.write(f"**CeRA:** correct={bool(row['cera_correct'])}, extracted={row['cera_extracted']}\n\n")
            handle.write(f"**LoRA:** correct={bool(row['lora_correct'])}, extracted={row['lora_extracted']}\n\n")
            handle.write("### Prompt\n\n```text\n" + row["prompt"] + "\n```\n\n")
            handle.write("### CeRA generation\n\n```text\n" + row["cera_generated"] + "\n```\n\n")
            handle.write("### LoRA generation\n\n```text\n" + row["lora_generated"] + "\n```\n\n")


def print_summary(seed_summaries, cross_seed):
    grader = "math_verify" if _HAVE_MV else "fallback"
    print(f"grading backend: {grader}")
    print("seed      N    CeRA    LoRA   delta       C-only/L-only  McNemar p      bootstrap 95% CI")
    for summary in seed_summaries:
        interval = summary["bootstrap_95_ci_percentage_points"]
        interval_text = "disabled" if interval is None else f"[{interval[0]:+.2f}, {interval[1]:+.2f}]"
        print(
            f"{summary['seed']:<7}{summary['n']:>5}"
            f"{100 * summary['cera']['accuracy']:>8.2f}"
            f"{100 * summary['lora']['accuracy']:>8.2f}"
            f"{summary['delta_percentage_points']:>+8.2f}"
            f"{summary['cera_only_correct']:>10}/{summary['lora_only_correct']:<7}"
            f"{summary['mcnemar_exact_p']:>10.4f}  {interval_text}"
        )
    if cross_seed:
        print("\nCross-seed (descriptive; repeated questions are not independent samples):")
        print(json.dumps(cross_seed, indent=2))


def main():
    args = parse_args()
    rng = random.Random(args.random_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_summaries = []
    rows_by_seed = {}
    seen_seeds = set()
    for seed, cera_path, lora_path in args.pair:
        if seed in seen_seeds:
            raise ValueError(f"Duplicate seed label: {seed}")
        seen_seeds.add(seed)
        summary, rows = analyze_seed(
            seed, cera_path, lora_path, args.grading, args.bootstrap, rng
        )
        seed_summaries.append(summary)
        rows_by_seed[seed] = rows

    cross_seed = None
    stable_cera = set()
    stable_lora = set()
    if len(rows_by_seed) > 1:
        cross_seed, stable_cera, stable_lora = cross_seed_summary(rows_by_seed)

    result = {
        "grading": args.grading,
        "community_grader_backend": "math_verify" if _HAVE_MV else "fallback",
        "bootstrap_iterations": args.bootstrap,
        "per_seed": seed_summaries,
        "cross_seed": cross_seed,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    write_disagreements(
        args.output_dir / "disagreements.jsonl", rows_by_seed, stable_cera, stable_lora
    )
    write_review(
        args.output_dir / "review.md",
        rows_by_seed,
        stable_cera,
        stable_lora,
        args.max_review_cases,
    )
    print_summary(seed_summaries, cross_seed)
    print(f"\nWrote analysis to {args.output_dir}")


if __name__ == "__main__":
    main()