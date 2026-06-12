#!/usr/bin/env python3
"""
analyze_svd.py -- Batch Singular Value Spectrum Analysis.

Two operating modes (mutually exclusive):

  --base_dir DIR
      Scan DIR for all experiment folders that match --dataset.

  --folders F1 F2 ...
      Analyse a hand-picked list of experiment folder paths.

Output: a JSON file containing one record per experiment with keys
  experiment_id, config, ckpt_used, spectrum.

Example -- scan mode:
    python analysis/analyze_svd.py \\
        --base_dir results --dataset math \\
        --output_json results/svd_spectra.json

Example -- specify mode:
    python analysis/analyze_svd.py \\
        --folders results/Exp_CeRA_math_... results/Exp_LoRA_math_... \\
        --dataset math --output_json results/svd_ablation.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cera.metrics import find_best_checkpoint, get_singular_values, scan_experiment_dirs
from cera.data import load_task_dataset


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

DEFAULT_MODEL  = "meta-llama/Llama-3.1-8B"
_HF_TOKEN_RE   = re.compile(r"^hf_[A-Za-z0-9]{10,}$")
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE          = torch.bfloat16
DEFAULT_LAYERS = "27,28,29,30,31"
NUM_SAMPLES    = 128


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch SVD spectrum analysis of CeRA / LoRA adapter activations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--base_dir", default=None,
        help="[Scan mode] Root results/ directory to scan for experiment folders.",
    )
    mode.add_argument(
        "--folders", nargs="+", default=None,
        help="[Specify mode] Explicit list of experiment folder paths to analyse.",
    )

    p.add_argument(
        "--dataset", choices=["math", "code", "orca"], default="math",
        help="Dataset used to generate activation profiling inputs.",
    )
    p.add_argument(
        "--base_model", default=DEFAULT_MODEL,
        help="HuggingFace model identifier.",
    )
    p.add_argument(
        "--target_layers", default=DEFAULT_LAYERS,
        help="Comma-separated transformer layer indices to analyse.",
    )
    p.add_argument(
        "--output_json", default="results/svd_spectra.json",
        help="Output JSON file path.",
    )

    return p.parse_args()


# ------------------------------------------------------------------------------
# Per-folder processing
# ------------------------------------------------------------------------------

def process_folder(
    folder_path: Path,
    inputs: torch.Tensor,
    hf_token: Optional[str],
    base_model: str,
    target_layers: list,
) -> Optional[dict]:
    """Process one experiment folder and return a result dict or None."""
    m_type = None
    for t in ("CeRA", "LoRA"):
        if (folder_path / t).exists():
            m_type = t
            break

    if m_type is None:
        print(f"[SKIP] {folder_path.name}: no CeRA/ or LoRA/ subfolder.")
        return None

    sub_dir  = folder_path / m_type
    log_file = sub_dir / f"{m_type}_log.json"

    if not log_file.exists():
        print(f"[SKIP] {folder_path.name}: log file not found.")
        return None

    try:
        with open(log_file) as f:
            log_data = json.load(f)
        config = log_data.get("config", {})
    except Exception as exc:
        print(f"[ERROR] {folder_path.name}: cannot read log: {exc}")
        return None

    # Checkpoints trained on a different base model must be skipped: their
    # adapter keys do not match the injected structure, load_state_dict
    # (strict=False) silently loads nothing, and the resulting spectrum is
    # that of a randomly initialized adapter.
    trained_on = config.get("model", "")
    if trained_on and trained_on != base_model:
        print(f"[SKIP] {folder_path.name}: trained on {trained_on}, not {base_model}.")
        return None

    ckpt = find_best_checkpoint(sub_dir, m_type)
    if ckpt is None:
        print(f"[SKIP] {folder_path.name}: no checkpoint files found.")
        return None

    print(f" -> {folder_path.name} | {m_type} | ckpt={ckpt.name}")

    spectrum = get_singular_values(
        ckpt_path     = str(ckpt),
        inputs        = inputs,
        config        = config,
        model_name    = base_model,
        hf_token      = hf_token,
        layer_indices = target_layers,
        device        = DEVICE,
        dtype         = DTYPE,
    )

    if spectrum is None:
        return None

    return {
        "experiment_id": folder_path.name,
        "config":        config,
        "ckpt_used":     ckpt.name,
        "spectrum":      spectrum,
    }


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    args = parse_args()

    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN is not set.")
        sys.exit(1)
    if not _HF_TOKEN_RE.match(HF_TOKEN):
        print("[ERROR] HF_TOKEN format is invalid. Expected format: hf_<alphanumeric>")
        sys.exit(1)

    if args.folders is None and args.base_dir is None:
        print("[ERROR] Provide either --folders or --base_dir.")
        sys.exit(1)

    target_layers = [int(x.strip()) for x in args.target_layers.split(",")]

    # -- Profiling inputs -----------------------------------------------------
    print(f"[INFO] Loading {NUM_SAMPLES} profiling samples from '{args.dataset}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    ids_train, _ = load_task_dataset(args.dataset, tokenizer, max_train_samples=NUM_SAMPLES)
    inputs = ids_train[:NUM_SAMPLES]

    # -- Collect folder list ---------------------------------------------------
    if args.folders:
        folder_paths = [Path(f) for f in args.folders]
        print(f"[INFO] Specify mode: {len(folder_paths)} folder(s) to process.")
    else:
        found        = scan_experiment_dirs(args.base_dir, args.dataset)
        folder_paths = [fp for fp, _ in found]

    # -- Process ---------------------------------------------------------------
    results = []
    for fp in tqdm(folder_paths, desc="Analysing"):
        entry = process_folder(fp, inputs, HF_TOKEN, args.base_model, target_layers)
        if entry is not None:
            results.append(entry)

    # -- Save -----------------------------------------------------------------
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n[SUCCESS] {len(results)} spectra saved -> {args.output_json}")


if __name__ == "__main__":
    main()
