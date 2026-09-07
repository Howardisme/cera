#!/usr/bin/env python3
"""
evaluate.py -- Downstream Task Evaluation: Pass@k Exact Match on math benchmarks.

Loads a CausalLM with a CeRA, LoRA, or DoRA adapter from a training checkpoint
and evaluates exact-match accuracy on GSM8K, MathInstruct (held-out tail), or
the competition-level MATH dataset (DigitalLearningGmbH/MATH-lighteval).

When --num_samples_per_problem 1 (default), greedy decoding is used and only
Pass@1 is reported.

When --num_samples_per_problem N > 1, sampling is used (controlled by
--temperature and --top_p), and the unbiased Pass@k estimator from Chen et al.
2021 is reported for k = 1 .. N.

Unbiased estimator:
    pass@k = mean_i [ 1 - C(n-c_i, k) / C(n, k) ]
where n = num_samples_per_problem and c_i = correct samples for problem i.

Example (Pass@1, greedy, GSM8K):
    python evaluate.py \\
        --adapter_type cera --rank 128 --dropout 0.1 \\
        --checkpoint results/.../CeRA/cera_ckpt_best_75000.pt \\
        --dataset gsm8k \\
        --output_jsonl results/eval_cera_r128_gsm8k.jsonl

Example (Pass@10, sampling, MATH):
    python evaluate.py \\
        --adapter_type cera --rank 128 --dropout 0.1 \\
        --checkpoint results/.../CeRA/cera_ckpt_best_75000.pt \\
        --dataset math \\
        --num_samples_per_problem 10 --temperature 0.8 --top_p 0.95 \\
        --max_new_tokens 1024 \\
        --output_jsonl results/eval_passk_cera_r128_math.jsonl
"""

import argparse
import json
import os
import re
import sys
from math import comb
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cera.adapters import apply_cera, apply_lora, apply_dora


DEFAULT_MODEL         = "meta-llama/Llama-3.1-8B"
DEVICE                = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE                 = torch.bfloat16
_VALID_TARGET_MODULES = {"q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
_HF_TOKEN_RE          = re.compile(r"^hf_[A-Za-z0-9]{10,}$")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pass@k Exact Match evaluation for CeRA / LoRA / DoRA on math benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Base model --
    p.add_argument(
        "--base_model", default=DEFAULT_MODEL,
        help="HuggingFace model ID. atten_dim auto-detected from model.config.hidden_size.",
    )

    # -- Adapter --
    p.add_argument(
        "--adapter_type", choices=["cera", "lora", "dora"], required=True,
        help="Adapter architecture to evaluate.",
    )
    p.add_argument(
        "--rank", type=int, required=True,
        help="CeRA: controls expansion_factor = rank / hidden_size. LoRA/DoRA: bottleneck dim.",
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to the adapter checkpoint.  With --adapter_format legacy "
             "(default) this is a .pt file produced by cera.trainer.save_checkpoint. "
             "With --adapter_format peft this is the directory containing "
             "adapter_model.safetensors + adapter_config.json produced by "
             "train_peft.py (e.g. results/.../peft_adapter_best_<step>).",
    )
    p.add_argument(
        "--adapter_format", choices=["legacy", "peft"], default="legacy",
        help="Checkpoint format.  'legacy' loads a .pt state_dict into a model "
             "wrapped with cera.adapters.apply_{cera,lora,dora}.  'peft' loads "
             "a PEFT adapter directory via peft.PeftModel.from_pretrained; the "
             "adapter architecture is read from adapter_config.json, so --rank / "
             "--alpha / --target_modules are ignored in that mode (they are still "
             "accepted for CLI-shape compatibility with legacy runs).  CeRA is "
             "not supported in 'peft' mode -- train_peft.py's CeRA path also "
             "produces legacy .pt files.",
    )
    p.add_argument(
        "--alpha", type=int, default=32,
        help="LoRA/DoRA alpha scaling factor (ignored for CeRA).",
    )
    p.add_argument(
        "--act_fn", choices=["silu", "relu", "identity"], default="silu",
        help="CeRA activation function -- must match the training config.",
    )
    p.add_argument(
        "--dropout", type=float, default=0.1,
        help="CeRA dropout value -- architecture must match training (inactive during eval).",
    )
    p.add_argument(
        "--target_modules", default="q_proj,v_proj",
        help="Comma-separated attention projections -- must match training config.",
    )

    # -- Dataset --
    p.add_argument(
        "--dataset", choices=["gsm8k", "mathinstruct", "math", "math500", "math_hard"], default="gsm8k",
        help=(
            "Benchmark to evaluate on. "
            "'math' = DigitalLearningGmbH/MATH-lighteval (5000 test problems, "
            "'Problem:/Solution:' prompt -- matches paper Table 1). "
            "'math500' = HuggingFaceH4/MATH-500 (community-standard 500-problem "
            "subset, 'Question:/Answer:' prompt -- aligned with MathInstruct "
            "training format, breaks paper Table 1 comparability by design). "
            "'math_hard' = MATH-lighteval test filtered to level=='Level 5' "
            "(~1324 hardest-tier Hendrycks MATH problems, 'Question:/Answer:' "
            "prompt -- harder difficulty point between MATH-500 and AMC/AIME). "
            "Recommend --max_new_tokens 1024 for math/math500/math_hard."
        ),
    )
    p.add_argument(
        "--num_samples", type=int, default=None,
        help="Cap the number of evaluation problems (default: full split).",
    )

    # -- Sampling / Pass@k --
    p.add_argument(
        "--num_samples_per_problem", type=int, default=1,
        help=(
            "Number of independent generations per problem. "
            "1 = greedy Pass@1 (default). "
            "N > 1 enables sampling and reports unbiased Pass@1..N."
        ),
    )
    p.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature (only used when num_samples_per_problem > 1).",
    )
    p.add_argument(
        "--top_p", type=float, default=0.95,
        help="Nucleus sampling top-p (only used when num_samples_per_problem > 1).",
    )

    # -- Generation --
    p.add_argument("--batch_size",     type=int, default=4,
                   help="Number of problems per batch.")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--repetition_penalty", type=float, default=1.0,
                   help="Decode-time repetition penalty (1.0 = off). Use ~1.15 to "
                        "suppress greedy repetition-collapse.")

    # -- Output --
    p.add_argument(
        "--output_jsonl", default="results/eval_results.jsonl",
        help="Path for the per-sample JSONL results file.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_with_adapter(args: argparse.Namespace, hf_token: str):
    target_modules = [t.strip() for t in args.target_modules.split(",")]
    invalid = [m for m in target_modules if m not in _VALID_TARGET_MODULES]
    if invalid:
        print(f"[ERROR] Unknown target_modules: {invalid}. Allowed: {sorted(_VALID_TARGET_MODULES)}")
        sys.exit(1)

    if args.adapter_format == "peft" and args.adapter_type == "cera":
        print("[ERROR] adapter_format='peft' is only valid for LoRA / DoRA. "
              "CeRA checkpoints are always legacy .pt (train_peft.py routes "
              "CeRA through cera.adapters.apply_cera, not through PEFT).")
        sys.exit(1)

    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-padding required for batched generation

    print("[INFO] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        token=hf_token,
        torch_dtype=DTYPE,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = True
    for param in model.parameters():
        param.requires_grad = False

    # atten_dim auto-detected so the same rank value works across model sizes
    atten_dim = model.config.hidden_size
    print(f"[INFO] atten_dim={atten_dim} (from model.config.hidden_size)")

    if args.adapter_format == "peft":
        # PEFT-managed adapter: architecture is read from adapter_config.json,
        # so --rank / --alpha / --target_modules from the CLI are ignored here.
        from peft import PeftModel

        if not os.path.isdir(args.checkpoint):
            print(f"[ERROR] --adapter_format peft expects a directory containing "
                  f"adapter_model.safetensors + adapter_config.json, got: {args.checkpoint}")
            sys.exit(1)

        print(f"[INFO] Loading PEFT adapter directory: {args.checkpoint}")
        model = PeftModel.from_pretrained(model, args.checkpoint, is_trainable=False)
        print(f"[INFO] PEFT adapter loaded (config from adapter_config.json).")

    else:
        print(f"[INFO] Injecting {args.adapter_type.upper()} adapter (rank={args.rank})...")
        if args.adapter_type == "cera":
            model = apply_cera(
                model,
                expansion_factor = args.rank / atten_dim,
                dropout          = args.dropout,
                act_fn           = args.act_fn,
                target_modules   = target_modules,
            )
        elif args.adapter_type == "dora":
            model = apply_dora(
                model,
                rank           = args.rank,
                alpha          = args.alpha,
                target_modules = target_modules,
            )
        else:  # lora
            model = apply_lora(
                model,
                rank           = args.rank,
                alpha          = args.alpha,
                dropout        = args.dropout,
                target_modules = target_modules,
            )

        print(f"[INFO] Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(
            f"[INFO] Checkpoint loaded."
            f" Missing keys: {len(result.missing_keys)}"
            f" | Unexpected keys: {len(result.unexpected_keys)}"
        )

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_eval_dataset(args: argparse.Namespace) -> List[Dict]:
    """Returns a list of dicts: {"prompt": str, "gold_raw": str}"""
    print(f"[INFO] Loading dataset: {args.dataset}...")

    if args.dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        samples = [
            {
                "prompt":   f"Question: {s['question']}\nAnswer:",
                "gold_raw": s["answer"].split("####")[-1].strip(),
            }
            for s in ds
        ]

    elif args.dataset == "mathinstruct":
        ds = load_dataset("TIGER-Lab/MathInstruct", split="train")
        start  = 100_000
        end    = start + 2_000
        subset = ds.select(range(start, min(end, len(ds))))
        samples = [
            {
                "prompt":   f"Question: {s['instruction']}\nAnswer:",
                "gold_raw": s["output"],
            }
            for s in subset
        ]

    elif args.dataset == "math500":
        # HuggingFaceH4/MATH-500: community-standard 500-problem subset of Hendrycks
        # MATH. Uses 'Question:/Answer:' prompt to align with MathInstruct training
        # format (see cera/data.py:fmt_math) -- deliberate departure from paper
        # Table 1 to remove the train/eval prompt-distribution-shift confound.
        # The 'answer' field is the pre-extracted final answer string (no \boxed{}
        # wrapping), so we skip extract_boxed for the gold side downstream.
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        samples = [
            {
                "prompt":   f"Question: {s['problem']}\nAnswer:",
                "gold_raw": s["answer"],
            }
            for s in ds
        ]

    elif args.dataset == "math_hard":
        # MATH-lighteval test set filtered to Level 5 (hardest tier of Hendrycks
        # MATH). ~1324 problems, harder than MATH-500 average, easier than AIME.
        # Uses Q/A prompt to align with training. Gold answer extracted from
        # solution via extract_boxed downstream (MATH-lighteval has no separate
        # 'answer' field, only 'solution' with \boxed{} inside).
        ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")
        ds = ds.filter(lambda s: s.get("level") == "Level 5")
        samples = [
            {
                "prompt":   f"Question: {s['problem']}\nAnswer:",
                "gold_raw": s["solution"],
            }
            for s in ds
        ]

    else:  # math (competition-level, DigitalLearningGmbH/MATH-lighteval)
        ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")
        samples = [
            {
                "prompt":   f"Problem: {s['problem']}\nSolution:",
                "gold_raw": s["solution"],
            }
            for s in ds
        ]

    if args.num_samples is not None:
        samples = samples[: args.num_samples]

    print(f"[INFO] {len(samples)} evaluation problems ready.")
    return samples


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> Optional[str]:
    """
    Extract the content of the LAST \\boxed{...} in text using brace-balanced
    scanning.  Handles arbitrarily nested curly braces.
    Returns None if no \\boxed{ is found or braces are unbalanced.
    """
    key = r'\boxed{'
    idx = text.rfind(key)
    if idx == -1:
        return None
    start = idx + len(key)
    depth = 1
    i     = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start : i - 1]


_FALLBACK_PATTERNS = [
    r'####\s*(-?[\d,\.\/]+)',
    r'[Tt]he answer is[:\s]+(-?[\d,\.\/\w]+)',
    r'[Ff]inal [Aa]nswer[:\s]+(-?[\d,\.\/\w]+)',
    r'(?m)^Answer:\s*(-?[\d,\.\/]+)\s*$',
    r'(-?[\d,]+\.?\d*)\s*$',
]


def extract_answer(text: str) -> Optional[str]:
    """Extract answer from generated text. Priority: \\boxed{} > fallback patterns."""
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    for pattern in _FALLBACK_PATTERNS:
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()
    return None


_WRAP_CMD_RE = re.compile(
    r'\\(?:text|textbf|textit|mathbf|mathrm|mathit|boldsymbol)\{([^{}]*)\}'
)


def normalize_latex(s: Optional[str]) -> str:
    """Normalize a LaTeX math string for Exact Match on the MATH dataset."""
    if s is None:
        return ""
    s = s.strip()
    prev = None
    while prev != s:
        prev = s
        s = _WRAP_CMD_RE.sub(r'\1', s)
    s = re.sub(r'\\frac\s*([^{\s])\s*([^{\s])', r'\\frac{\1}{\2}', s)
    s = re.sub(r'\s+', '', s)
    return s


def normalize(s: Optional[str]) -> str:
    """Normalize a numeric answer string for GSM8K / MathInstruct."""
    if s is None:
        return ""
    s = s.strip()
    s = s.replace(",", "")
    s = s.replace("$", "")
    s = s.replace("%", "")
    s = s.rstrip(".")
    s = s.lower()
    return s.strip()


# ---------------------------------------------------------------------------
# Pass@k estimator
# ---------------------------------------------------------------------------

def _pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimator (Chen et al. 2021, Appendix A):
        pass@k = 1 - C(n-c, k) / C(n, k)
    """
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def compute_pass_at_k(
    correct_counts: List[int],
    n: int,
    max_k: int,
) -> Dict[int, float]:
    """Return {k: pass@k} for k = 1 .. max_k."""
    results = {}
    for k in range(1, max_k + 1):
        if k > n:
            break
        vals = [_pass_at_k(n, c, k) for c in correct_counts]
        results[k] = float(np.mean(vals))
    return results


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def _batched(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def run_evaluation(
    model,
    tokenizer,
    samples: List[Dict],
    args: argparse.Namespace,
) -> List[Dict]:
    """Run evaluation and return per-problem records."""
    n_per_prob   = args.num_samples_per_problem
    use_sampling = n_per_prob > 1
    records      = []

    for batch in tqdm(list(_batched(samples, args.batch_size)), desc="Evaluating"):
        prompts   = [s["prompt"]   for s in batch]
        golds_raw = [s["gold_raw"] for s in batch]

        enc = tokenizer(
            prompts,
            return_tensors = "pt",
            padding        = True,
            truncation     = True,
            max_length     = 512,
        ).to(DEVICE)

        prompt_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens      = args.max_new_tokens,
                do_sample           = use_sampling,
                temperature         = args.temperature if use_sampling else 1.0,
                top_p               = args.top_p       if use_sampling else 1.0,
                num_return_sequences= n_per_prob,
                pad_token_id        = tokenizer.eos_token_id,
                eos_token_id        = tokenizer.eos_token_id,
                repetition_penalty  = args.repetition_penalty,
            )

        gen_tokens = out_ids[:, prompt_len:]
        generated  = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        is_math_comp = args.dataset in ("math", "math500", "math_hard")
        _normalize   = normalize_latex if is_math_comp else normalize

        for prob_idx, (prompt, gold_raw) in enumerate(zip(prompts, golds_raw)):
            gens_for_prob = generated[prob_idx * n_per_prob : (prob_idx + 1) * n_per_prob]

            if args.dataset in ("math", "math_hard"):
                # MATH-lighteval solutions embed the gold answer in \boxed{...}.
                extracted_gold = extract_boxed(gold_raw)
            elif args.dataset == "math500":
                # HuggingFaceH4/MATH-500 'answer' field is already the extracted
                # final answer string (no \boxed{} wrapping) -- use as-is.
                extracted_gold = gold_raw
            elif args.dataset == "mathinstruct":
                extracted_gold = extract_answer(gold_raw)
            else:
                extracted_gold = gold_raw

            gold_norm = _normalize(extracted_gold)

            sample_records = []
            for gen_text in gens_for_prob:
                extracted_pred = extract_answer(gen_text)
                pred_norm      = _normalize(extracted_pred)
                correct        = (pred_norm == gold_norm) and (pred_norm != "")
                sample_records.append({
                    "generated":        gen_text,
                    "extracted_answer": extracted_pred,
                    "correct":          correct,
                })

            records.append({
                "id":           len(records),
                "dataset":      args.dataset,
                "prompt":       prompt,
                "gold_answer":  extracted_gold,
                "samples":      sample_records,
                "n_correct":    sum(s["correct"] for s in sample_records),
            })

    return records


def print_summary(records: List[Dict], args: argparse.Namespace):
    n_per_prob = args.num_samples_per_problem
    n_problems = len(records)
    correct_counts = [r["n_correct"] for r in records]

    print("\n" + "=" * 60)
    print(f"Dataset    : {args.dataset}  ({n_problems} problems)")
    print(
        f"Adapter    : {args.adapter_type.upper()}  R={args.rank}"
        f"  ckpt={Path(args.checkpoint).name}"
    )
    print(f"Samples/problem : {n_per_prob}")
    print("-" * 60)

    passk = compute_pass_at_k(correct_counts, n_per_prob, max_k=n_per_prob)
    for k, val in passk.items():
        print(f"  Pass@{k:<2d} : {val * 100:.2f}%")

    n_extraction_failures = sum(
        1
        for r in records
        for s in r["samples"]
        if s["extracted_answer"] is None
    )
    total_gens = n_problems * n_per_prob
    print("-" * 60)
    print(
        f"Extraction failures : {n_extraction_failures} / {total_gens}"
        f"  ({n_extraction_failures / total_gens * 100:.1f}%)"
    )
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Legacy .pt is a file; PEFT adapter is a directory containing
    # adapter_model.safetensors + adapter_config.json.
    if args.adapter_format == "peft":
        if not os.path.isdir(args.checkpoint):
            print(f"[ERROR] --adapter_format peft expects a directory, got: {args.checkpoint}")
            sys.exit(1)
    else:
        if not os.path.isfile(args.checkpoint):
            print(f"[ERROR] Checkpoint file not found: {args.checkpoint}")
            sys.exit(1)

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("[ERROR] HF_TOKEN is not set. Check your .env file.")
        sys.exit(1)
    if not _HF_TOKEN_RE.match(hf_token):
        print("[ERROR] HF_TOKEN format is invalid. Expected format: hf_<alphanumeric>")
        sys.exit(1)

    model, tokenizer = load_model_with_adapter(args, hf_token)
    samples          = load_eval_dataset(args)
    records          = run_evaluation(model, tokenizer, samples, args)

    print_summary(records, args)

    out_path = Path(args.output_jsonl).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[INFO] Results saved -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
