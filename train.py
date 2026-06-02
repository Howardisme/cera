#!/usr/bin/env python3
"""
train.py -- Canonical entry point for CeRA / LoRA / DoRA fine-tuning.

Fine-tunes a HuggingFace CausalLM (default: Llama-3.1-8B) with a parameter-
efficient adapter. Supports CeRA (non-linear parallel branch), LoRA (low-rank
linear), and DoRA (weight-decomposed LoRA).

Example:
    python train.py \\
        --model_type CeRA  --rank 128 --lr 5e-4 --dropout 0.1 \\
        --dataset math --epochs 3

Example (ablation -- different activation or target modules):
    python train.py \\
        --model_type CeRA --rank 128 --lr 5e-4 \\
        --act_fn relu --target_modules q_proj,v_proj,o_proj \\
        --dataset orca --epochs 3
"""

import argparse
import datetime
import gc
import os
import re
import sys

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from cera.adapters import apply_cera, apply_lora, apply_dora
from cera.data import load_task_dataset, load_forgetting_dataset
from cera.trainer import (
    CHECKPOINT_DATA_SIZES,
    setup_dual_logger,
    run_experiment,
)


# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

DEFAULT_MODEL    = "meta-llama/Llama-3.1-8B"
_VALID_TARGET_MODULES = {"q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
_HF_TOKEN_RE     = re.compile(r"^hf_[A-Za-z0-9]{10,}$")
DEVICE           = "cuda"
DTYPE            = torch.bfloat16
BATCH_SIZE       = 4
GRAD_ACCUM_STEPS = 16            # Effective batch size = 4 * 16 = 64


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune a CausalLM with CeRA, LoRA, or DoRA adapters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Base model -----------------------------------------------------------
    p.add_argument(
        "--base_model", default=DEFAULT_MODEL,
        help="HuggingFace model ID. atten_dim is auto-detected from model.config.hidden_size.",
    )

    # -- Adapter --------------------------------------------------------------
    p.add_argument(
        "--model_type", choices=["CeRA", "LoRA", "DoRA"], default="CeRA",
        help="Adapter architecture.",
    )
    p.add_argument(
        "--rank", type=int, default=64,
        help=(
            "For LoRA/DoRA: low-rank bottleneck dimension.  "
            "For CeRA: controls expansion_factor = rank / hidden_size."
        ),
    )
    p.add_argument(
        "--alpha", type=int, default=32,
        help="LoRA/DoRA scaling factor alpha. Effective scale = alpha / rank.",
    )
    p.add_argument(
        "--dropout", type=float, default=0.1,
        help="Structural dropout probability inside the CeRA bottleneck (CeRA only).",
    )
    p.add_argument(
        "--act_fn", choices=["silu", "relu", "identity"], default="silu",
        help="Non-linearity for the CeRA up-projection (CeRA only).",
    )
    p.add_argument(
        "--target_modules", default="q_proj,v_proj",
        help="Comma-separated attention projections to adapt.",
    )

    # -- Training -------------------------------------------------------------
    p.add_argument(
        "--dataset", choices=["math", "code", "orca"], default="math",
        help="Target fine-tuning dataset.",
    )
    p.add_argument(
        "--epochs", type=int, default=3,
        help="Number of full passes over the training set.",
    )
    p.add_argument(
        "--lr", type=float, default=5e-4,
        help="AdamW learning rate.",
    )

    return p.parse_args()


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    args = parse_args()

    # -- Environment ----------------------------------------------------------
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN is not set. Check your .env file or environment.")
        sys.exit(1)
    if not _HF_TOKEN_RE.match(HF_TOKEN):
        print("[ERROR] HF_TOKEN format is invalid. Expected format: hf_<alphanumeric>")
        sys.exit(1)

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"]   = "120"

    target_modules = [t.strip() for t in args.target_modules.split(",")]
    invalid = [m for m in target_modules if m not in _VALID_TARGET_MODULES]
    if invalid:
        print(f"[ERROR] Unknown target_modules: {invalid}. Allowed: {sorted(_VALID_TARGET_MODULES)}")
        sys.exit(1)

    # -- Output paths ---------------------------------------------------------
    WORK_DIR  = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    tgt_str   = args.target_modules.replace(",", "_")
    model_tag = args.base_model.split("/")[-1]
    model_suffix = f"_{model_tag}" if args.base_model != DEFAULT_MODEL else ""
    exp_name  = (
        f"Exp_{args.model_type}_{args.dataset}"
        f"_R{args.rank}_lr{args.lr}_{args.act_fn}_{tgt_str}"
        f"_D{args.dropout}_E{args.epochs}{model_suffix}_{timestamp}"
    )
    base_save = os.path.join(WORK_DIR, "results", exp_name)
    results_root = os.path.realpath(os.path.join(WORK_DIR, "results"))
    if not os.path.realpath(base_save).startswith(results_root + os.sep):
        print(f"[ERROR] Resolved save path escapes results directory: {os.path.realpath(base_save)}")
        sys.exit(1)
    save_dir  = os.path.join(base_save, args.model_type)
    os.makedirs(save_dir, exist_ok=True)

    # -- Dual logging ---------------------------------------------------------
    setup_dual_logger(os.path.join(base_save, "console_output.txt"))

    print(f"[INFO] Results -> {base_save}")
    print(
        f"[INFO] Device: "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    print(
        f"[INFO] Config: model={args.model_type} | dataset={args.dataset} | "
        f"rank={args.rank} | lr={args.lr} | dropout={args.dropout} | "
        f"act_fn={args.act_fn} | targets={target_modules} | epochs={args.epochs}"
    )

    # -- Tokenizer ------------------------------------------------------------
    print("\n[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    # -- Datasets -------------------------------------------------------------
    ids_train_target, ids_test_target = load_task_dataset(
        args.dataset, tokenizer, max_train_samples=100_000
    )
    ids_train_orig, ids_test_orig = load_forgetting_dataset(tokenizer)
    gc.collect()

    # -- Base model -----------------------------------------------------------
    print("\n[INFO] Loading base model (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        token=HF_TOKEN,
        torch_dtype=DTYPE,
        device_map="auto",
        attn_implementation="sdpa",
        resume_download=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    for p in model.parameters():
        p.requires_grad = False

    # expansion_factor for CeRA: rank / hidden_size (auto-detected per model)
    atten_dim = model.config.hidden_size
    cera_exp  = args.rank / atten_dim
    print(f"[INFO] atten_dim={atten_dim} | cera_exp={cera_exp:.6f}")

    # -- Adapter injection ----------------------------------------------------
    if args.model_type == "CeRA":
        model = apply_cera(
            model, cera_exp,
            dropout=args.dropout,
            act_fn=args.act_fn,
            target_modules=target_modules,
        )
    elif args.model_type == "DoRA":
        model = apply_dora(
            model, args.rank,
            alpha=args.alpha,
            target_modules=target_modules,
        )
    else:  # LoRA
        model = apply_lora(
            model, args.rank,
            alpha=args.alpha,
            dropout=args.dropout,
            target_modules=target_modules,
        )

    model.to(DEVICE)

    # -- Config dict (written into checkpoint logs) ---------------------------
    config = {
        "model":          args.base_model,
        "model_type":     args.model_type,
        "dataset":        args.dataset,
        "rank":           args.rank,
        "lr":             args.lr,
        "dropout":        args.dropout,
        "act_fn":         args.act_fn  if args.model_type == "CeRA" else "linear",
        "alpha":          args.alpha   if args.model_type in ("LoRA", "DoRA") else None,
        "target_modules": args.target_modules,
        "epochs":         args.epochs,
        "grad_accum":     GRAD_ACCUM_STEPS,
        "batch_size":     BATCH_SIZE,
    }

    # -- Run ------------------------------------------------------------------
    run_experiment(
        model            = model,
        ids_train_target = ids_train_target,
        ids_test_target  = ids_test_target,
        ids_train_orig   = ids_train_orig,
        ids_test_orig    = ids_test_orig,
        config           = config,
        save_dir         = save_dir,
        batch_size       = BATCH_SIZE,
        grad_accum_steps = GRAD_ACCUM_STEPS,
        device           = DEVICE,
        pad_token_id     = tokenizer.pad_token_id,
    )

    print("\n[SUCCESS] Training complete.")


if __name__ == "__main__":
    main()
