#!/usr/bin/env python3
"""
train_peft.py -- HuggingFace Trainer + PEFT + TRL pipeline for CeRA / LoRA / DoRA.

Companion to train.py.  The command-line interface, experiment tag, and
output layout are kept identical so that a run of train_peft.py can be
compared directly against the corresponding run of train.py; the only
difference is the training pipeline, which is delegated to standard
HuggingFace tooling.

Pipeline differences relative to train.py
-----------------------------------------
  * Adapter injection for LoRA / DoRA uses peft.LoraConfig via
    get_peft_model.  CeRA continues to use cera.adapters.apply_cera as PEFT
    provides no non-linear adapter primitive.
  * Training loop is trl.SFTTrainer (a thin subclass of transformers.Trainer).
  * Batch sampling is a shuffled DataLoader (without replacement).  train.py
    uses numpy.random.choice with the default replace=True, so within one
    nominal epoch the same sample can be drawn multiple times and others
    never seen.
  * Loss is restricted to answer tokens via DataCollatorForCompletionOnlyLM.
    train.py includes the entire prompt in the loss.
  * Learning-rate schedule is cosine with a 3 % warmup ratio.  train.py
    holds the learning rate constant.
  * Optimizer is adamw_torch_fused.  train.py uses eager AdamW.
  * Attention implementation is flash_attention_2 when the package is
    importable; otherwise sdpa (matching train.py).

Output compatibility
--------------------
  * Metrics JSON is written to <save_dir>/<model_type>_log.json in the same
    schema as cera.trainer.run_experiment, so downstream analysis scripts
    that consume the log continue to work unchanged.
  * The adapter is saved in PEFT's native format (adapter_model.safetensors
    + adapter_config.json) under <save_dir>/peft_adapter_<step>/ at each
    CHECKPOINT_DATA_SIZES threshold; this is the canonical checkpoint for
    LoRA / DoRA runs produced by this script.  A legacy .pt containing the
    raw trainable parameters is also written for CeRA runs (which do not
    go through PEFT), matching cera.trainer.save_checkpoint exactly.
  * evaluate.py currently loads legacy .pt via apply_lora / apply_dora and
    strict=False; loading a PEFT-format adapter requires an additional code
    path (peft.PeftModel.from_pretrained).  This is out of scope here.

Added dependencies (append to requirements.txt): peft>=0.11, trl>=0.11
"""

import argparse
import datetime
import gc
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    set_seed as hf_set_seed,
)

from peft import LoraConfig, get_peft_model
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from cera.adapters import apply_cera
from cera.data import (
    MAX_SEQ_LEN,
    METAMATH_TEST_N,
    METAMATH_TEST_OFF,
    METAMATH_TRAIN_N,
    fmt_code,
    fmt_math,
    fmt_metamathqa,
    fmt_orca,
    load_forgetting_dataset,
    load_task_dataset,
)
from cera.trainer import (
    CHECKPOINT_DATA_SIZES,
    run_eval,
    save_checkpoint,
    save_logs,
    setup_dual_logger,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL          = "meta-llama/Llama-3.1-8B"
_VALID_TARGET_MODULES  = {"q_proj", "v_proj", "k_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"}
_HF_TOKEN_RE           = re.compile(r"^hf_[A-Za-z0-9]{10,}$")
DEVICE                 = "cuda"
DTYPE                  = torch.bfloat16
BATCH_SIZE             = 4
GRAD_ACCUM_STEPS       = 16      # Effective batch size = 4 * 16 = 64
DEFAULT_SEED           = 42
DEFAULT_LR             = 1e-4
WARMUP_RATIO           = 0.03
LR_SCHEDULE            = "cosine"

# The response template must tokenise identically in isolation and in
# context; a leading newline avoids the BPE-merge boundary issue that
# otherwise makes DataCollatorForCompletionOnlyLM fail to locate the
# marker in some sequences.  All formatting functions in cera.data emit
# "Answer:" on a new line, so this template matches every task except
# fmt_orca (which uses "Assistant:"; see _response_template_for).
_RESPONSE_TEMPLATE_QA  = "\nAnswer:"
_RESPONSE_TEMPLATE_CHAT = "\nAssistant:"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


# ---------------------------------------------------------------------------
# Text-form dataset loader
#
# SFTTrainer expects raw strings + tokenizer, whereas cera.data returns
# pre-tokenised tensors for use with the numpy.random.choice loop in
# cera.trainer.run_experiment.  This helper mirrors load_task_dataset but
# yields text so that the formatting helpers remain the single source of
# truth for how each dataset is rendered into a training sequence.
# ---------------------------------------------------------------------------

def _load_text_dataset(
    dataset_name: str,
    tokenizer,
    max_train_samples: int,
) -> Tuple[Dataset, Dataset]:
    eos = tokenizer.eos_token

    if dataset_name == "math":
        ds = load_dataset("TIGER-Lab/MathInstruct", split="train")
        n_needed  = min(100_000 + 2_000, len(ds))
        rows      = list(ds.select(range(n_needed)))
        train_raw = rows[:max_train_samples]
        test_raw  = rows[100_000:]
        fmt = lambda x: fmt_math(x, eos)

    elif dataset_name == "metamathqa":
        ds = load_dataset("meta-math/MetaMathQA", split="train")
        train_cap = min(max_train_samples, METAMATH_TRAIN_N)
        n_needed  = min(METAMATH_TEST_OFF + METAMATH_TEST_N, len(ds))
        rows      = list(ds.select(range(n_needed)))
        train_raw = rows[:train_cap]
        test_raw  = rows[METAMATH_TEST_OFF:]
        fmt = lambda x: fmt_metamathqa(x, eos)

    elif dataset_name == "code":
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        n     = len(ds)
        split = int(n * 0.9)
        train_raw = list(ds.select(range(split)))
        test_raw  = list(ds.select(range(split, n)))
        fmt = lambda x: fmt_code(x, eos)

    elif dataset_name == "orca":
        ds = load_dataset("Open-Orca/SlimOrca", split="train")
        n_needed  = min(100_000 + 2_000, len(ds))
        rows      = list(ds.select(range(n_needed)))
        train_raw = rows[:max_train_samples]
        test_raw  = rows[100_000:]
        fmt = lambda x: fmt_orca(x, eos)

    else:
        raise ValueError(f"Unknown dataset {dataset_name!r}")

    train_texts = [fmt(x) for x in train_raw]
    test_texts  = [fmt(x) for x in test_raw]
    print(f"[DATA] Text-form dataset {dataset_name}: "
          f"{len(train_texts):,} train | {len(test_texts):,} test")

    return (
        Dataset.from_dict({"text": train_texts}),
        Dataset.from_dict({"text": test_texts}),
    )


def _response_template_for(dataset_name: str) -> str:
    return _RESPONSE_TEMPLATE_CHAT if dataset_name == "orca" else _RESPONSE_TEMPLATE_QA


# ---------------------------------------------------------------------------
# Adapter injection
# ---------------------------------------------------------------------------

def _apply_peft_lora(
    model,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: List[str],
    use_dora: bool,
):
    """Wrap `model` with a PEFT LoraConfig (use_dora=True for DoRA)."""
    cfg = LoraConfig(
        r              = rank,
        lora_alpha     = alpha,
        lora_dropout   = dropout,
        target_modules = target_modules,
        bias           = "none",
        task_type      = "CAUSAL_LM",
        use_dora       = use_dora,
    )
    return get_peft_model(model, cfg)


# ---------------------------------------------------------------------------
# Periodic-eval callback
#
# Mirrors the eval + checkpoint cadence of cera.trainer.run_experiment so
# that a train_peft.py run produces a metrics JSON with the same set of
# entries as a train.py run (indexed by data_seen), for direct A/B analysis.
# ---------------------------------------------------------------------------

class ForgettingEvalCallback(TrainerCallback):
    def __init__(
        self,
        model,
        ids_test_target: torch.Tensor,
        ids_test_orig: torch.Tensor,
        save_dir: str,
        model_type: str,
        config: Dict[str, Any],
        checkpoint_sizes: List[int],
        effective_batch: int,
        pad_token_id: int,
        peft_managed: bool,
    ):
        self.model            = model
        self.ids_test_target  = ids_test_target
        self.ids_test_orig    = ids_test_orig
        self.save_dir         = save_dir
        self.model_type       = model_type
        self.log              = {"config": config, "history": []}
        self.checkpoint_sizes = sorted(set(checkpoint_sizes))
        self.effective_batch  = effective_batch
        self.pad_token_id     = pad_token_id
        self.peft_managed     = peft_managed
        self._fired: set      = set()
        self._best_val_loss   = float("inf")
        self._best_ckpt_dir: Optional[str] = None
        self._best_ckpt_path: Optional[str] = None
        self._running_loss    = 0.0
        self._running_steps   = 0

    def _persist_checkpoint(self, step: int, data_seen: int, record: Dict) -> Tuple[str, Optional[str]]:
        """Save an adapter checkpoint; return (native_dir_or_none, legacy_pt_path)."""
        native_dir: Optional[str] = None
        legacy_path: Optional[str] = None

        if self.peft_managed:
            native_dir = os.path.join(self.save_dir, f"peft_adapter_{step}")
            self.model.save_pretrained(native_dir)
            print(f"[SAVE] PEFT adapter -> {native_dir}")
        else:
            legacy_fname = f"{self.model_type.lower()}_ckpt_{data_seen}.pt"
            save_checkpoint(self.model, step, data_seen, record,
                            self.save_dir, self.model_type,
                            filename=legacy_fname)
            legacy_path = os.path.join(self.save_dir, legacy_fname)

        return native_dir, legacy_path

    def _run_and_log(self, state, data_seen: int):
        step = state.global_step
        avg_tr = (self._running_loss / self._running_steps
                  if self._running_steps else float("nan"))
        self._running_loss, self._running_steps = 0.0, 0

        print(f"\n[EVAL] step={step} | data_seen={data_seen:,}")
        lo_te, pp_te = run_eval(
            self.model, self.ids_test_target, BATCH_SIZE, DEVICE,
            self.pad_token_id,
        )
        lo_or, pp_or = run_eval(
            self.model, self.ids_test_orig, BATCH_SIZE, DEVICE,
            self.pad_token_id,
        )

        record = {
            "step":       step,
            "data_seen":  data_seen,
            "new_task_target": {
                "train_loss": avg_tr,
                "train_ppl":  float(np.exp(avg_tr)) if avg_tr < 20 else float("inf"),
                "test_loss":  lo_te,
                "test_ppl":   pp_te,
            },
            "orig_task_general": {"test_loss": lo_or, "test_ppl": pp_or},
        }
        self.log["history"].append(record)
        save_logs(self.log, self.save_dir, self.model_type)
        print(f"  target_ppl={pp_te:.2f} | general_ppl={pp_or:.2f}")

        # Threshold checkpoint (matches CHECKPOINT_DATA_SIZES semantics).
        self._persist_checkpoint(step, data_seen, record)

        # Best-checkpoint tracker mirrors cera.trainer.run_experiment.
        if lo_te < self._best_val_loss:
            self._best_val_loss = lo_te

            # Remove the previous best to keep disk usage bounded.
            if self._best_ckpt_dir and os.path.isdir(self._best_ckpt_dir):
                import shutil
                shutil.rmtree(self._best_ckpt_dir, ignore_errors=True)
            if self._best_ckpt_path and os.path.exists(self._best_ckpt_path):
                os.remove(self._best_ckpt_path)

            if self.peft_managed:
                best_dir = os.path.join(self.save_dir, f"peft_adapter_best_{step}")
                self.model.save_pretrained(best_dir)
                self._best_ckpt_dir  = best_dir
                self._best_ckpt_path = None
                print(f"[SAVE] New best val_loss={self._best_val_loss:.4f} -> {best_dir}")
            else:
                best_fname = f"{self.model_type.lower()}_ckpt_best_{step}.pt"
                save_checkpoint(self.model, step, data_seen, record,
                                self.save_dir, self.model_type,
                                filename=best_fname)
                self._best_ckpt_dir  = None
                self._best_ckpt_path = os.path.join(self.save_dir, best_fname)
                print(f"[SAVE] New best val_loss={self._best_val_loss:.4f}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Trainer emits "loss" at logging_steps intervals; accumulate for the
        # avg-train-loss field of the next EVAL record.
        if logs and "loss" in logs:
            self._running_loss  += float(logs["loss"])
            self._running_steps += 1

    def on_step_end(self, args, state, control, **kwargs):
        data_seen = state.global_step * self.effective_batch
        for size in self.checkpoint_sizes:
            if size in self._fired:
                continue
            if data_seen >= size:
                self._fired.add(size)
                self._run_and_log(state, size)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune a CausalLM with CeRA / LoRA / DoRA using the "
                    "HuggingFace Trainer + PEFT + TRL pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Base model -----------------------------------------------------------
    p.add_argument("--base_model", default=DEFAULT_MODEL,
                   help="HuggingFace model ID.")

    # -- Adapter --------------------------------------------------------------
    p.add_argument("--model_type", choices=["CeRA", "LoRA", "DoRA"], default="LoRA")
    p.add_argument("--rank",    type=int,   default=64)
    p.add_argument("--alpha",   type=int,   default=64,
                   help="Scale-matched default (alpha == rank).")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="LoRA/DoRA input dropout, or CeRA bottleneck dropout.")
    p.add_argument("--act_fn",  choices=["silu", "relu", "identity"], default="silu",
                   help="CeRA only.")
    p.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,"
                                                "gate_proj,up_proj,down_proj",
                   help="Comma-separated target projections (default: all_linear).")

    # -- Training -------------------------------------------------------------
    p.add_argument("--dataset",  choices=["math", "metamathqa", "code", "orca"],
                   default="metamathqa")
    p.add_argument("--epochs",   type=int,   default=3)
    p.add_argument("--lr",       type=float, default=DEFAULT_LR)
    p.add_argument("--seed",     type=int,   default=DEFAULT_SEED)
    p.add_argument("--optim",    default="adamw_torch_fused",
                   help="Trainer optim key; fall back to 'adamw_torch' if fused kernels "
                        "are unavailable.")
    p.add_argument("--attn_impl", choices=["flash_attention_2", "sdpa"], default="sdpa",
                   help="Set to flash_attention_2 if the flash-attn package is installed.")
    p.add_argument("--logging_steps", type=int, default=25)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # -- Environment ----------------------------------------------------------
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN is not set.")
        sys.exit(1)
    if not _HF_TOKEN_RE.match(HF_TOKEN):
        print("[ERROR] HF_TOKEN format is invalid.")
        sys.exit(1)

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"]   = "120"

    set_seed(args.seed)

    target_modules = [t.strip() for t in args.target_modules.split(",")]
    invalid = [m for m in target_modules if m not in _VALID_TARGET_MODULES]
    if invalid:
        print(f"[ERROR] Unknown target_modules: {invalid}. "
              f"Allowed: {sorted(_VALID_TARGET_MODULES)}")
        sys.exit(1)

    # -- Output paths ---------------------------------------------------------
    WORK_DIR  = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    tgt_str   = args.target_modules.replace(",", "_")
    model_tag = args.base_model.split("/")[-1]
    model_suffix = f"_{model_tag}" if args.base_model != DEFAULT_MODEL else ""
    alpha_suffix = (
        f"_A{args.alpha}"
        if args.model_type in ("LoRA", "DoRA") and args.alpha != 32
        else ""
    )
    seed_suffix = f"_S{args.seed}" if args.seed != DEFAULT_SEED else ""
    exp_name  = (
        f"Exp_PEFT_{args.model_type}_{args.dataset}"
        f"_R{args.rank}_lr{args.lr}_{args.act_fn}_{tgt_str}"
        f"_D{args.dropout}_E{args.epochs}{alpha_suffix}{seed_suffix}{model_suffix}_{timestamp}"
    )
    base_save    = os.path.join(WORK_DIR, "results", exp_name)
    results_root = os.path.realpath(os.path.join(WORK_DIR, "results"))
    if not os.path.realpath(base_save).startswith(results_root + os.sep):
        print(f"[ERROR] Resolved save path escapes results directory.")
        sys.exit(1)
    save_dir = os.path.join(base_save, args.model_type)
    os.makedirs(save_dir, exist_ok=True)

    setup_dual_logger(os.path.join(base_save, "console_output.txt"))

    print(f"[INFO] Results -> {base_save}")
    print(f"[INFO] Device: "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"[INFO] Config: pipeline=PEFT | model={args.model_type} | "
          f"dataset={args.dataset} | rank={args.rank} | alpha={args.alpha} | "
          f"lr={args.lr} | dropout={args.dropout} | act_fn={args.act_fn} | "
          f"targets={target_modules} | epochs={args.epochs} | seed={args.seed}")

    # -- Tokenizer ------------------------------------------------------------
    print("\n[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # -- Datasets -------------------------------------------------------------
    # Text form for SFTTrainer training loop.
    train_text_ds, _ = _load_text_dataset(
        args.dataset, tokenizer, max_train_samples=100_000
    )
    # Pre-tokenised tensors reused for periodic eval (run_eval expects tensors,
    # unchanged from cera.trainer, so log entries stay directly comparable).
    _, ids_test_target = load_task_dataset(
        args.dataset, tokenizer, max_train_samples=100_000
    )
    _, ids_test_orig = load_forgetting_dataset(tokenizer)
    gc.collect()

    # -- Base model -----------------------------------------------------------
    print("\n[INFO] Loading base model (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        token             = HF_TOKEN,
        torch_dtype       = DTYPE,
        device_map         = "auto",
        attn_implementation = args.attn_impl,
        resume_download    = True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    for p in model.parameters():
        p.requires_grad = False

    # -- Adapter injection ----------------------------------------------------
    peft_managed = args.model_type in ("LoRA", "DoRA")
    if args.model_type == "CeRA":
        atten_dim = model.config.hidden_size
        cera_exp  = args.rank / atten_dim
        print(f"[INFO] atten_dim={atten_dim} | cera_exp={cera_exp:.6f}")
        model = apply_cera(
            model, cera_exp,
            dropout        = args.dropout,
            act_fn         = args.act_fn,
            target_modules = target_modules,
        )
    else:
        use_dora = (args.model_type == "DoRA")
        print(f"[INFO] Applying PEFT LoRA | rank={args.rank} | alpha={args.alpha} | "
              f"dropout={args.dropout} | targets={target_modules} | use_dora={use_dora}")
        model = _apply_peft_lora(
            model,
            rank           = args.rank,
            alpha          = args.alpha,
            dropout        = args.dropout,
            target_modules = target_modules,
            use_dora       = use_dora,
        )
        model.print_trainable_parameters()

    # -- Config dict (mirrors train.py log schema) ----------------------------
    config = {
        "pipeline":       "peft",
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
        "seed":           args.seed,
        "optim":          args.optim,
        "lr_scheduler":   LR_SCHEDULE,
        "warmup_ratio":   WARMUP_RATIO,
        "attn_impl":      args.attn_impl,
        "response_template": _response_template_for(args.dataset),
        "loss_scope":     "completion_only",
    }

    # -- SFTTrainer -----------------------------------------------------------
    # packing=False is intentional: DataCollatorForCompletionOnlyLM is
    # incompatible with sequence packing, and correct instruction-tuning
    # loss (answer tokens only) is prioritised over the packing throughput
    # win.  Packing can be evaluated as a separate ablation.
    sft_cfg = SFTConfig(
        output_dir                   = save_dir,
        num_train_epochs             = args.epochs,
        per_device_train_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps  = GRAD_ACCUM_STEPS,
        learning_rate                = args.lr,
        lr_scheduler_type            = LR_SCHEDULE,
        warmup_ratio                 = WARMUP_RATIO,
        optim                        = args.optim,
        max_grad_norm                = 1.0,
        bf16                         = True,
        gradient_checkpointing       = True,
        gradient_checkpointing_kwargs= {"use_reentrant": False},
        logging_steps                = args.logging_steps,
        save_strategy                = "no",       # Checkpointing handled by the callback.
        report_to                    = "none",
        seed                         = args.seed,
        data_seed                    = args.seed,
        dataset_text_field           = "text",
        max_seq_length               = MAX_SEQ_LEN,
        packing                      = False,
        remove_unused_columns        = False,
    )

    response_template = _response_template_for(args.dataset)
    collator = DataCollatorForCompletionOnlyLM(
        response_template = response_template,
        tokenizer         = tokenizer,
    )

    effective_batch = BATCH_SIZE * GRAD_ACCUM_STEPS
    callback = ForgettingEvalCallback(
        model            = model,
        ids_test_target  = ids_test_target,
        ids_test_orig    = ids_test_orig,
        save_dir         = save_dir,
        model_type       = args.model_type,
        config           = config,
        checkpoint_sizes = CHECKPOINT_DATA_SIZES,
        effective_batch  = effective_batch,
        pad_token_id     = tokenizer.pad_token_id,
        peft_managed     = peft_managed,
    )

    trainer = SFTTrainer(
        model           = model,
        args            = sft_cfg,
        train_dataset   = train_text_ds,
        tokenizer       = tokenizer,
        data_collator   = collator,
        callbacks       = [callback],
    )

    # -- Baseline evaluation before any optimizer step ------------------------
    print("\n[STEP 0] Baseline Evaluation...")
    lo_te, pp_te = run_eval(model, ids_test_target, BATCH_SIZE, DEVICE,
                            tokenizer.pad_token_id)
    lo_or, pp_or = run_eval(model, ids_test_orig, BATCH_SIZE, DEVICE,
                            tokenizer.pad_token_id)
    baseline = {
        "step": 0, "data_seen": 0,
        "new_task_target":   {"train_loss": None, "train_ppl": None,
                              "test_loss":  lo_te, "test_ppl":  pp_te},
        "orig_task_general": {"test_loss":  lo_or, "test_ppl":  pp_or},
    }
    callback.log["history"].append(baseline)
    save_logs(callback.log, save_dir, args.model_type)
    print(f"  baseline target_ppl={pp_te:.2f} | general_ppl={pp_or:.2f}")

    # -- Run ------------------------------------------------------------------
    trainer.train()

    # -- Final checkpoint -----------------------------------------------------
    final_step = trainer.state.global_step
    final_data_seen = final_step * effective_batch
    final_record = callback.log["history"][-1] if callback.log["history"] else baseline
    if peft_managed:
        final_dir = os.path.join(save_dir, f"peft_adapter_final_{final_step}")
        model.save_pretrained(final_dir)
        print(f"[SAVE] Final PEFT adapter -> {final_dir}")
    else:
        final_fname = f"{args.model_type.lower()}_ckpt_final_{final_step}.pt"
        save_checkpoint(model, final_step, final_data_seen, final_record,
                        save_dir, args.model_type, filename=final_fname)

    print("\n[SUCCESS] Training complete.")


if __name__ == "__main__":
    main()
