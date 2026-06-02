"""
Training loop, checkpointing, and logging utilities for CeRA experiments.

Public API
----------
  DualLogger           : tee stdout/stderr to a file simultaneously
  setup_dual_logger()  : redirect sys.stdout/stderr to a DualLogger
  save_checkpoint()    : persist adapter weights + metrics to disk
  save_logs()          : write experiment history JSON
  run_eval()           : compute average loss + perplexity on a dataset
  run_experiment()     : full training loop with periodic eval + checkpointing
"""

import gc
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Default data-count checkpoints (number of training samples seen)
CHECKPOINT_DATA_SIZES: List[int] = [1_000, 5_000, 10_000, 25_000, 50_000, 75_000, 100_000]


# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------

class DualLogger:
    """
    Tee: simultaneously writes to the terminal and a persistent log file.

    Usage:
        sys.stdout = DualLogger("path/to/console_output.txt")
        sys.stderr = sys.stdout
    """

    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def setup_dual_logger(log_path: str) -> DualLogger:
    """Redirect sys.stdout and sys.stderr to a DualLogger and return it."""
    logger = DualLogger(log_path)
    sys.stdout = logger
    sys.stderr = logger
    return logger


# ------------------------------------------------------------------------------
# Checkpointing
# ------------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    step: int,
    data_count: int,
    metrics: Dict,
    save_dir: str,
    model_type: str,
    max_retries: int = 3,
    filename: Optional[str] = None,
):
    """
    Save only the adapter's trainable parameters plus experiment metadata.

    If filename is provided it is used directly; otherwise falls back to the
    legacy pattern {model_type.lower()}_ckpt_{data_count}.pt.
    """
    fname = filename or f"{model_type.lower()}_ckpt_{data_count}.pt"
    if os.sep in fname or (os.altsep and os.altsep in fname):
        raise ValueError(f"Invalid checkpoint filename contains path separator: {fname!r}")
    print(f"[SAVE] {model_type} checkpoint -> {fname}")
    state = {k: v.cpu() for k, v in model.named_parameters() if v.requires_grad}
    path  = os.path.join(save_dir, fname)

    for attempt in range(max_retries):
        try:
            torch.save(
                {
                    "step":             step,
                    "data_count":       data_count,
                    "model_type":       model_type,
                    "model_state_dict": state,
                    "metrics":          metrics,
                },
                path,
            )
            print(f"[SAVE] OK: {path}")
            return
        except OSError as exc:
            print(f"[SAVE] Attempt {attempt + 1}/{max_retries} failed: {exc}")
            time.sleep(2)

    print(f"[SAVE] All {max_retries} attempts failed -- checkpoint NOT saved.")


def save_logs(logs: Dict, save_dir: str, model_type: str):
    """Overwrite the experiment JSON log with the full history."""
    path = os.path.join(save_dir, f"{model_type}_log.json")
    with open(path, "w") as f:
        json.dump(logs, f, indent=4)
    print(f"[LOG] Updated -> {path}")


# ------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------

def run_eval(
    model: nn.Module,
    dataset_ids: torch.Tensor,
    batch_size: int,
    device: str,
    pad_token_id: int,
    limit_batches: int = 200,
) -> Tuple[float, float]:
    """
    Compute average cross-entropy loss and perplexity on a tokenised dataset.

    Args:
        model:         Model in train mode (temporarily switched to eval).
        dataset_ids:   Token-ID tensor [N, seq_len].
        batch_size:    Evaluation mini-batch size.
        device:        Target device string.
        pad_token_id:  ID to mask out in labels (typically eos_token_id).
        limit_batches: Maximum number of batches to evaluate (for speed).

    Returns:
        (avg_loss, perplexity)
    """
    model.eval()
    total_loss, cnt = 0.0, 0
    limit = min(len(dataset_ids), limit_batches * batch_size)

    with torch.no_grad():
        for i in range(0, limit, batch_size):
            if i + batch_size > len(dataset_ids):
                break
            batch  = dataset_ids[i : i + batch_size].to(device)
            labels = batch.clone()
            labels[labels == pad_token_id] = -100
            total_loss += model(batch, labels=labels).loss.item()
            cnt += 1

    model.train()
    model.enable_input_require_grads()

    if cnt == 0:
        return float("inf"), float("inf")

    avg_loss = total_loss / cnt
    ppl = float(np.exp(avg_loss)) if avg_loss < 20 else float("inf")
    return avg_loss, ppl


# ------------------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------------------

def run_experiment(
    model: nn.Module,
    ids_train_target: torch.Tensor,
    ids_test_target:  torch.Tensor,
    ids_train_orig:   torch.Tensor,
    ids_test_orig:    torch.Tensor,
    config: Dict[str, Any],
    save_dir: str,
    batch_size:        int = 4,
    grad_accum_steps:  int = 16,
    device:            str = "cuda",
    pad_token_id:      int = 2,
    checkpoint_sizes:  Optional[List[int]] = None,
):
    """
    Full training loop with periodic evaluation and checkpointing.

    The model is expected to already have adapter weights injected and base
    weights frozen before this call.  The function deletes the model and
    optimizer on completion to free GPU memory.

    Args:
        model:             Adapted model on the target device.
        ids_train_target:  Training tokens for the target task.
        ids_test_target:   Test tokens for the target task.
        ids_train_orig:    WikiText training tokens (forgetting metric).
        ids_test_orig:     WikiText test tokens (forgetting metric).
        config:            Metadata dict written into checkpoint logs.
                           Required keys: 'model_type', 'lr', 'epochs'.
        save_dir:          Directory for .pt checkpoints and JSON log.
        batch_size:        Mini-batch size (default 4).
        grad_accum_steps:  Steps before an optimizer update (effective batch
                           = batch_size * grad_accum_steps = 64 by default).
        device:            Training device ('cuda' or 'cpu').
        pad_token_id:      Token ID masked to -100 in labels.
        checkpoint_sizes:  Data-seen counts at which to checkpoint.
                           Defaults to CHECKPOINT_DATA_SIZES.
    """
    os.makedirs(save_dir, exist_ok=True)

    if checkpoint_sizes is None:
        checkpoint_sizes = CHECKPOINT_DATA_SIZES

    model_type = config["model_type"]
    lr         = config["lr"]
    epochs     = config.get("epochs", 1)

    # Cap training data
    max_train = 100_000
    if len(ids_train_target) > max_train:
        ids_train_target = ids_train_target[:max_train]

    full_data  = len(ids_train_target)
    steps_per  = full_data // batch_size
    max_steps  = steps_per * epochs

    # Build step -> data_seen checkpoint map
    ckpt_map = {size: size // batch_size for size in checkpoint_sizes}
    ckpt_map[full_data * epochs] = max_steps                       # always ckpt at end
    ckpt_map = {k: v for k, v in ckpt_map.items() if v <= max_steps}
    check_steps = sorted(set(ckpt_map.values()))

    # Summary
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print("\n" + "=" * 60)
    print(
        f"[START] {model_type} | dataset={config.get('dataset','?')} | "
        f"data={full_data:,} | epochs={epochs} | steps={max_steps:,}"
    )
    print(f"[STATS] Trainable: {trainable:,}  ({100 * trainable / total_p:.4f}%)")
    print("=" * 60)

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    log: Dict[str, Any] = {"config": config, "history": []}

    # -- Baseline evaluation --------------------------------------------------
    print("\n[STEP 0] Baseline Evaluation...")
    lo_tr, pp_tr   = run_eval(model, ids_train_target, batch_size, device, pad_token_id, 50)
    lo_te, pp_te   = run_eval(model, ids_test_target,  batch_size, device, pad_token_id)
    lo_or, pp_or   = run_eval(model, ids_test_orig,    batch_size, device, pad_token_id)

    m0 = {
        "step": 0, "data_seen": 0,
        "new_task_target":   {"train_loss": lo_tr, "train_ppl": pp_tr,
                              "test_loss":  lo_te, "test_ppl":  pp_te},
        "orig_task_general": {"test_loss":  lo_or, "test_ppl":  pp_or},
    }
    log["history"].append(m0)

    # Initialize best-checkpoint tracker from baseline val loss
    best_val_loss  = lo_te
    best_ckpt_name = f"{model_type.lower()}_ckpt_best_0.pt"
    best_ckpt_path = os.path.join(save_dir, best_ckpt_name)
    save_checkpoint(model, 0, 0, m0, save_dir, model_type, filename=best_ckpt_name)
    save_logs(log, save_dir, model_type)

    # -- Training -------------------------------------------------------------
    model.train()
    print("\n[INFO] Training started...")
    running_loss, step_cnt = 0.0, 0
    optimizer.zero_grad()

    # Track last eval metrics for the final checkpoint
    last_step, last_data_seen, last_m = 0, 0, m0

    try:
        for step in range(1, max_steps + 1):
            idx   = np.random.choice(full_data, batch_size)
            batch = ids_train_target[idx].to(device)
            labels = batch.clone()
            labels[labels == pad_token_id] = -100

            loss = model(batch, labels=labels).loss / grad_accum_steps
            loss.backward()
            running_loss += loss.item() * grad_accum_steps
            step_cnt     += 1

            if step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % 200 == 0:
                print(
                    f"  step {step:>6}/{max_steps} | "
                    f"loss {loss.item() * grad_accum_steps:.4f}"
                )

            if step in check_steps:
                # Flush any partial accumulation
                if step % grad_accum_steps != 0:
                    optimizer.step()
                    optimizer.zero_grad()

                data_seen = [k for k, v in ckpt_map.items() if v == step][0]
                avg_tr    = running_loss / step_cnt
                running_loss, step_cnt = 0.0, 0

                print(f"\n[EVAL] step={step} | data_seen={data_seen:,}")
                lo_te, pp_te = run_eval(model, ids_test_target, batch_size, device, pad_token_id)
                lo_or, pp_or = run_eval(model, ids_test_orig,   batch_size, device, pad_token_id)

                m = {
                    "step": step, "data_seen": data_seen,
                    "new_task_target": {
                        "train_loss": avg_tr,
                        "train_ppl":  float(np.exp(avg_tr)),
                        "test_loss":  lo_te,
                        "test_ppl":   pp_te,
                    },
                    "orig_task_general": {"test_loss": lo_or, "test_ppl": pp_or},
                }
                log["history"].append(m)
                print(f"  target_ppl={pp_te:.2f} | general_ppl={pp_or:.2f}")
                save_logs(log, save_dir, model_type)

                # Best checkpoint: save only when val loss improves
                if lo_te < best_val_loss:
                    best_val_loss = lo_te
                    if os.path.exists(best_ckpt_path):
                        os.remove(best_ckpt_path)
                        print(f"[SAVE] Deleted previous best: {os.path.basename(best_ckpt_path)}")
                    best_ckpt_name = f"{model_type.lower()}_ckpt_best_{step}.pt"
                    best_ckpt_path = os.path.join(save_dir, best_ckpt_name)
                    save_checkpoint(model, step, data_seen, m, save_dir, model_type,
                                    filename=best_ckpt_name)
                    print(f"[SAVE] New best val_loss={best_val_loss:.4f}")
                else:
                    print(
                        f"[SAVE] No improvement "
                        f"(val_loss={lo_te:.4f} >= best={best_val_loss:.4f}), skipping."
                    )

                last_step, last_data_seen, last_m = step, data_seen, m

        # Final checkpoint after all epochs complete
        final_fname = f"{model_type.lower()}_ckpt_final_{step}.pt"
        save_checkpoint(model, step, last_data_seen, last_m, save_dir, model_type,
                        filename=final_fname)
        print(f"[SAVE] Final checkpoint saved: {final_fname}")

    except Exception as exc:
        print(f"\n[CRITICAL ERROR] {model_type} training crashed: {exc}")
        save_logs(log, save_dir, model_type)
        raise

    finally:
        del model, optimizer
        torch.cuda.empty_cache()
        gc.collect()
