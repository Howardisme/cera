#!/usr/bin/env python3
"""
Community-consensus offline re-grade of existing eval jsonl.

Fixes the two eval-hygiene problems in evaluate.py, WITHOUT re-running the model:
  (B1) extraction: prefer \\boxed{} (last) > first "#### X" > "the answer is X";
       DROP the fragile bare-"last number at end of string" fallback that grabbed
       garbage from degenerate tails like ".3.3.3".
  (B2) grading: math-equivalence via `math_verify` (community standard, used by
       lighteval / Open LLM Leaderboard) if installed, else a numeric + hardened
       LaTeX string comparison that handles \\left\\right, \\dfrac, 1/2 vs 0.5,
       whitespace, $ … $, \\text{…}, degrees — the classes evaluate.py's thin
       normalize_latex missed (e.g. `(3,\\frac{\\pi}{2})` vs `\\left(3,\\frac{\\pi}{2}\\right)`).

*** CRITICAL LIMITATION ***
This re-grades the ALREADY-GENERATED text. It fixes grading false-negatives
(which affect all methods, roughly symmetrically) and recovers a few
"stated the answer then looped" cases. It CANNOT fix degeneration: an output
that looped into a WRONG repeated answer (LoRA's dominant failure mode here)
stays wrong. The differential-degeneration confound (LoRA/DoRA loop far more
than CeRA) is a DECODING problem — the real fix is to RE-GENERATE with
`repetition_penalty` / explicit eos / a stop sequence (see run_eval patch notes).
Use this script to measure how much of the gap is GRADING artifact; use
re-generation to remove the DECODING artifact.

Usage (run in the container so math_verify/その依存 is importable, else it
falls back to the dependency-free grader):
  python3 analysis/regrade_community.py \
      results/eval_outputs/r64_cera_lr1e-4_mm_all/math500_pass1.jsonl \
      results/eval_outputs/r64_lora_a64_lr1e-4_mm_all/math500_pass1.jsonl \
      results/eval_outputs/r64_dora_a64_lr1e-4_mm_all/math500_pass1.jsonl
  # no args -> globs results/eval_outputs/*/math500_pass1.jsonl
"""
import json, re, sys, glob, os

# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
def last_boxed(text):
    key = r'\boxed{'
    idx = text.rfind(key)
    if idx == -1:
        return None
    i = idx + len(key); depth = 1; j = i
    while j < len(text) and depth:
        depth += (text[j] == '{') - (text[j] == '}')
        j += 1
    return text[i:j-1] if depth == 0 else None

# FIRST occurrence of each pattern = the answer as stated BEFORE any repetition loop.
_ANS_PATTS = [
    r'####\s*([^\n#]+?)\s*(?:\n|####|$)',
    r'[Tt]he\s+answer\s+is[:\s]*\$?\s*([^\n.$]+?)\s*\$?\s*(?:\.|\n|$)',
    r'[Ff]inal\s+answer[:\s]*\$?\s*([^\n.$]+?)\s*\$?\s*(?:\.|\n|$)',
    r'(?m)^Answer:\s*([^\n]+?)\s*$',
]
def extract(text):
    if not text:
        return None
    b = last_boxed(text)
    if b is not None:
        return b.strip()
    for p in _ANS_PATTS:
        m = re.search(p, text)         # search = FIRST match => pre-degeneration
        if m:
            return m.group(1).strip()
    return None                         # NO bare-last-number fallback (grabbed loop garbage)

# --------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------
_HAVE_MV = False
try:
    from math_verify import parse as _mv_parse, verify as _mv_verify
    _HAVE_MV = True
except Exception:
    _HAVE_MV = False

def _hard_norm(s):
    if s is None:
        return ""
    s = str(s).strip()
    for a, b in [(r'\left', ''), (r'\right', ''), (r'\!', ''), (r'\,', ''),
                 (r'\;', ''), (r'\ ', ''), (r'\dfrac', r'\frac'), (r'\tfrac', r'\frac'),
                 ('$', ''), ('%', ''), (r'^\circ', ''), (r'^{\circ}', ''), (r'\cdot', '*')]:
        s = s.replace(a, b)
    s = re.sub(r'\\text\w*\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\mbox\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\frac\s*([^{\s])\s*([^{\s])', r'\\frac{\1}{\2}', s)
    s = re.sub(r'\s+', '', s)
    return s.rstrip('.').lower()

def _to_num(s):
    """Best-effort numeric value (handles \\frac{a}{b}, a/b, decimals)."""
    if s is None:
        return None
    t = _hard_norm(s)
    m = re.fullmatch(r'\\frac\{(-?\d+)\}\{(-?\d+)\}', t) or re.fullmatch(r'(-?\d+)/(-?\d+)', t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return a / b if b else None
    try:
        return float(t)
    except Exception:
        return None

def grade(pred, gold):
    if pred is None or gold is None:
        return False
    if _HAVE_MV:
        try:
            return bool(_mv_verify(_mv_parse(str(gold)), _mv_parse(str(pred))))
        except Exception:
            pass
    pn, gn = _to_num(pred), _to_num(gold)
    if pn is not None and gn is not None:
        return abs(pn - gn) < 1e-6
    hp = _hard_norm(pred)
    return hp != "" and hp == _hard_norm(gold)

# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def regrade_file(path):
    n = old_c = new_c = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        gold = r.get('gold_answer')
        old = 1 if r.get('n_correct', 0) > 0 else 0
        new = 1 if any(grade(extract(s.get('generated', '')), gold)
                       for s in (r.get('samples') or [])) else 0
        n += 1; old_c += old; new_c += new
    return n, old_c, new_c

def main(paths):
    print(f"grader: {'math_verify (community standard)' if _HAVE_MV else 'numeric + hardened-string fallback (math_verify not importable)'}\n")
    print(f"{'cell':44s}{'N':>5}{'old%':>8}{'new%':>8}{'Δcorrect':>10}")
    print("-" * 75)
    for path in paths:
        name = os.path.basename(os.path.dirname(path)) or path
        n, oc, nc = regrade_file(path)
        print(f"{name:44s}{n:>5}{100*oc/n:>8.1f}{100*nc/n:>8.1f}{nc-oc:>+10d}")

if __name__ == '__main__':
    paths = sys.argv[1:] or sorted(glob.glob('results/eval_outputs/*/math500_pass1.jsonl'))
    main(paths)
