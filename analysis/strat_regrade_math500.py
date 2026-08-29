#!/usr/bin/env python3
"""
MATH-500 per-level stratification using the community re-grader (math_verify).

Shows whether a difficulty gradient (CeRA-LoRA delta rising with level) emerges
once grading false-negatives — which concentrate at high difficulty and penalize
CeRA more — are fixed. Prints OLD (stored) vs NEW (math-verify re-graded) per
level, with paired McNemar (CeRA vs LoRA) on the re-graded correctness.

Free (no GPU). Needs `datasets` (+ math_verify if available), so run inside the
container. Reuses extract/grade from analysis/regrade_community.py.

Run (from repo root, /work/$USER/cera):
  singularity exec -B /work --env PYTHONPATH=/work/$USER/cera_pypkgs \
    --env PYTHONNOUSERSITE=1 --env HF_HOME=/work/$USER/hf_cache /work/$USER/cera.sif \
    python3 analysis/strat_regrade_math500.py
"""
import sys, os, json
from math import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regrade_community import extract, grade, _HAVE_MV

CELLS = {
    "CeRA": "results/eval_outputs/r64_cera_lr1e-4_mm_all/math500_pass1.jsonl",
    "LoRA": "results/eval_outputs/r64_lora_a64_lr1e-4_mm_all/math500_pass1.jsonl",
    "DoRA": "results/eval_outputs/r64_dora_a64_lr1e-4_mm_all/math500_pass1.jsonl",
}


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def norm(v):
    if isinstance(v, int):
        return v
    for ch in str(v or ""):
        if ch.isdigit():
            return int(ch)
    return None


def mcnemar(pairs):
    B = sum(1 for a, b in pairs if a == 1 and b == 0)
    C = sum(1 for a, b in pairs if a == 0 and b == 1)
    nn = B + C
    if nn == 0:
        return B, C, 1.0
    k = min(B, C)
    return B, C, min(1.0, 2 * sum(comb(nn, i) for i in range(k + 1)) / 2 ** nn)


def main():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    lvl_by_id = [norm(r.get("level")) for r in ds]

    print("grader: %s\n" % ("math_verify" if _HAVE_MV else "numeric + hardened-string fallback"))

    data = {}   # name -> {id: (old_correct, new_correct)}
    for name, path in CELLS.items():
        d = {}
        for r in load(path):
            i = r.get("id")
            if i is None:
                continue
            old = 1 if r.get("n_correct", 0) > 0 else 0
            gold = r.get("gold_answer")
            new = 0
            for s in (r.get("samples") or []):
                if grade(extract(s.get("generated", "")), gold):
                    new = 1
                    break
            d[i] = (old, new)
        data[name] = d

    ids = sorted(set.intersection(*[set(d) for d in data.values()]))
    levels = sorted({lvl_by_id[i] for i in ids if lvl_by_id[i] is not None})

    def acc(name, sel, k):
        return 100.0 * sum(data[name][i][k] for i in sel) / len(sel) if sel else 0.0

    hdr = ("{:>3}{:>5}{:>7}{:>7}{:>7}   ||{:>7}{:>7}{:>7}{:>7}{:>8}{:>7}"
           .format("Lv", "N", "CeRA", "LoRA", "C-L", "CeRA", "LoRA", "DoRA", "C-L", "b/c", "McNp"))
    print("OLD grading (stored)      | NEW grading (math-verify) + McNemar(CeRA vs LoRA)")
    print(hdr)
    for L in levels:
        sub = [i for i in ids if lvl_by_id[i] == L]
        B, C, p = mcnemar([(data["CeRA"][i][1], data["LoRA"][i][1]) for i in sub])
        cl_old = acc("CeRA", sub, 0) - acc("LoRA", sub, 0)
        cl_new = acc("CeRA", sub, 1) - acc("LoRA", sub, 1)
        print("{:>3}{:>5}{:>7.1f}{:>7.1f}{:>+7.1f}   ||{:>7.1f}{:>7.1f}{:>7.1f}{:>+7.1f}{:>8}{:>7.3f}"
              .format(L, len(sub), acc("CeRA", sub, 0), acc("LoRA", sub, 0), cl_old,
                      acc("CeRA", sub, 1), acc("LoRA", sub, 1), acc("DoRA", sub, 1), cl_new,
                      "%d/%d" % (B, C), p))

    B, C, p = mcnemar([(data["CeRA"][i][1], data["LoRA"][i][1]) for i in ids])
    print("\noverall N=%d NEW: CeRA %.1f  LoRA %.1f  DoRA %.1f | C-L b/c=%d/%d p=%.3f"
          % (len(ids), acc("CeRA", ids, 1), acc("LoRA", ids, 1), acc("DoRA", ids, 1), B, C, p))


if __name__ == "__main__":
    main()
