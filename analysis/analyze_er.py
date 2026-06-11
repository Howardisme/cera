#!/usr/bin/env python3
"""
analyze_er.py -- Effective Rank (ER) trajectory analysis.

Computes the Effective Rank metric (Roy & Vetterli 2007) on adapter output
activations at multiple training checkpoints, showing how the adapter's
representational subspace evolves during fine-tuning.

Modes
-----
  manifold       [DEPRECATED] CeRA vs LoRA -- Manifold Expansion experiment
  lr_sensitivity [DEPRECATED] LoRA vs CeRA (low-LR) vs CeRA (high-LR)
  dropout        [DEPRECATED] CeRA (D=0) vs CeRA (D=0.3) -- Plasticity Control
  ablation       CeRA variants (activation fn, target modules, dropout)

DEPRECATION: the trajectory modes (manifold, lr_sensitivity, dropout) load
legacy per-data-count checkpoints ({type}_ckpt_{1000,10000,...}.pt), which the
current trainer no longer saves -- it keeps only *_ckpt_best_*.pt and
*_ckpt_final_*.pt. On new runs these modes find no checkpoints and emit an
empty CSV. For ER-vs-rank figures (paper Fig. 3/4), use:
    python analysis/plot_rank_scaling.py --metric er --svd_json <spectra.json>
which computes ER directly from the analyze_svd.py spectrum output.

Output: CSV rows to stdout  ->  redirect with  | tee results/er_manifold.csv

Example:
    python analysis/analyze_er.py --mode manifold \\
        --lora_path  results/.../LoRA \\
        --cera_path  results/.../CeRA

    python analysis/analyze_er.py --mode ablation \\
        --rank 128 \\
        --full_path      results/.../CeRA_Full/CeRA \\
        --module_path    results/.../CeRA_Mod/CeRA \\
        --linear_path    results/.../CeRA_Ident/CeRA \\
        --relu_path      results/.../CeRA_ReLU/CeRA \\
        --nodropout_path results/.../CeRA_NoDrop/CeRA
"""

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, NamedTuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cera.adapters import apply_cera, apply_lora
from cera.metrics import compute_effective_rank, compute_manifold_dim, find_best_checkpoint


# ------------------------------------------------------------------------------
# Constants (overridable via CLI)
# ------------------------------------------------------------------------------

DEFAULT_MODEL  = "meta-llama/Llama-3.1-8B"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE          = torch.bfloat16
DEFAULT_LAYERS = "27,28,29,30,31"   # last 5 layers of 32-layer Llama-3-8B
DEFAULT_SAMPLES = 128
CHECKPOINTS    = [1_000, 10_000, 50_000, 100_000]  # data-seen values to evaluate


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Effective Rank trajectory analysis across training checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode", required=True,
        choices=["manifold", "lr_sensitivity", "dropout", "ablation"],
        help="Analysis scenario.",
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
        "--num_samples", type=int, default=DEFAULT_SAMPLES,
        help="Number of MathInstruct samples for activation profiling.",
    )

    # Paths -- only relevant subset required per mode
    p.add_argument("--lora_path",      default=None, help="LoRA checkpoint directory.")
    p.add_argument("--cera_path",      default=None, help="CeRA checkpoint directory.")
    p.add_argument("--cera_low_path",  default=None,
                   help="[lr_sensitivity] CeRA directory for the low learning-rate run.")
    p.add_argument("--cera_high_path", default=None,
                   help="[lr_sensitivity] CeRA directory for the high learning-rate run.")
    p.add_argument("--no_dropout_path", default=None,
                   help="[dropout] CeRA dropout=0 checkpoint directory.")
    p.add_argument("--dropout_path",    default=None,
                   help="[dropout] CeRA dropout=0.3 checkpoint directory.")

    # Ablation mode paths
    p.add_argument("--full_path",      default=None, help="[ablation] CeRA full (SiLU, q+v, D=0.3)")
    p.add_argument("--module_path",    default=None, help="[ablation] Granularity variant (o_proj only)")
    p.add_argument("--linear_path",    default=None, help="[ablation] Identity activation variant")
    p.add_argument("--relu_path",      default=None, help="[ablation] ReLU activation variant")
    p.add_argument("--nodropout_path", default=None, help="[ablation] No-dropout variant (D=0.0)")

    # Overrides
    p.add_argument("--rank",    type=int,   default=None,
                   help="Adapter rank. Overrides the mode default.")
    p.add_argument("--dropout", type=float, default=None,
                   help="CeRA dropout. Overrides the mode default.")

    return p.parse_args()


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------

def load_profiling_inputs(base_model: str, num_samples: int) -> torch.Tensor:
    """Load a small MathInstruct subset for activation profiling."""
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    ds     = load_dataset("TIGER-Lab/MathInstruct", split="train")
    subset = ds.select(range(num_samples))

    def fmt(x):
        return f"Question: {x['instruction']}\nAnswer: {x['output']}{tokenizer.eos_token}"

    return tokenizer(
        [fmt(x) for x in subset],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )["input_ids"]


# ------------------------------------------------------------------------------
# Core analysis loop
# ------------------------------------------------------------------------------

# Config tuple: (display_label, ckpt_directory, model_type, rank, dropout)
ConfigList = List[Tuple[str, Optional[str], str, int, float]]


def _run_analysis(
    configs: ConfigList,
    inputs: torch.Tensor,
    base_model: str,
    target_layers: List[int],
):
    """
    Iterate over configurations and checkpoints, compute ER and
    Manifold Dimensionality, and print CSV rows to stdout.

    CSV header:  data_seen, label, effective_rank, manifold_dim
    """
    print(
        "# DEPRECATED: this trajectory mode needs legacy per-data-count "
        "checkpoints ({type}_ckpt_{N}.pt), which the current trainer no longer "
        "saves. New runs will produce no rows below. "
        "Use plot_rank_scaling.py --metric er instead.",
        file=sys.stderr,
    )
    print("data_seen,label,effective_rank,manifold_dim")

    target_modules = ["q_proj", "v_proj"]

    for label, ckpt_dir, m_type, rank, dropout in configs:
        if ckpt_dir is None:
            print(f"# WARNING: path not provided for '{label}', skipping.")
            continue

        for data_count in CHECKPOINTS:
            ckpt_name = f"{m_type.lower()}_ckpt_{data_count}.pt"
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            if not os.path.exists(ckpt_path):
                continue

            model = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=DTYPE, device_map=DEVICE
            )
            for p in model.parameters():
                p.requires_grad = False

            if m_type == "CeRA":
                atten_dim = model.config.hidden_size
                model = apply_cera(
                    model, rank / atten_dim,
                    dropout=dropout,
                    target_modules=target_modules,
                    layer_indices=target_layers,
                )
            else:
                model = apply_lora(
                    model, rank,
                    target_modules=target_modules,
                    layer_indices=target_layers,
                )

            model.to(device=DEVICE, dtype=DTYPE)

            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)

            # Enable activation tracking
            tracked = []
            for layer_idx in target_layers:
                layer = model.model.layers[layer_idx]
                for name in target_modules:
                    wrapper = getattr(layer.self_attn, name, None)
                    if wrapper is None:
                        continue
                    if m_type == "CeRA" and hasattr(wrapper, "cera"):
                        wrapper.cera.track_activation = True
                        tracked.append(wrapper.cera)
                    elif m_type == "LoRA" and hasattr(wrapper, "lora_A"):
                        wrapper.track_activation = True
                        tracked.append(wrapper)

            model.eval()
            with torch.no_grad():
                model(inputs.to(DEVICE))

            er_vals  = []
            dim_vals = []
            for mod in tracked:
                if mod.last_delta is None:
                    continue
                er_vals.append(compute_effective_rank(mod.last_delta))
                dim_vals.append(compute_manifold_dim(mod.last_delta))

            avg_er  = float(np.mean(er_vals))  if er_vals  else 0.0
            avg_dim = float(np.mean(dim_vals)) if dim_vals else 0.0
            print(f"{data_count},{label},{avg_er:.4f},{avg_dim:.2f}")

            del model, ckpt
            torch.cuda.empty_cache()
            gc.collect()


# ------------------------------------------------------------------------------
# Ablation ER analysis
# ------------------------------------------------------------------------------

class AblationSpec(NamedTuple):
    label: str
    ckpt_dir: Optional[str]
    act_fn: str
    target_modules: List[str]
    dropout: float


def _run_ablation_analysis(
    specs: List[AblationSpec],
    inputs: torch.Tensor,
    rank: int,
    base_model: str,
    target_layers: List[int],
):
    """Compute a single ER value per ablation variant at its best checkpoint."""
    print("variant,effective_rank")

    for spec in specs:
        if spec.ckpt_dir is None:
            print(f"# WARNING: path not provided for '{spec.label}', skipping.")
            continue

        ckpt_path = find_best_checkpoint(Path(spec.ckpt_dir), "CeRA")
        if ckpt_path is None:
            print(f"# WARNING: no checkpoint found in {spec.ckpt_dir}")
            continue

        print(
            f"[INFO] {spec.label} | act={spec.act_fn} | targets={spec.target_modules} "
            f"| D={spec.dropout} | ckpt={ckpt_path.name}",
            flush=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=DTYPE, device_map=DEVICE
        )
        for p in model.parameters():
            p.requires_grad = False

        atten_dim = model.config.hidden_size
        model = apply_cera(
            model, rank / atten_dim,
            dropout=spec.dropout,
            act_fn=spec.act_fn,
            target_modules=spec.target_modules,
            layer_indices=target_layers,
        )
        model.to(device=DEVICE, dtype=DTYPE)

        ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

        tracked = []
        for layer_idx in target_layers:
            layer = model.model.layers[layer_idx]
            for name in spec.target_modules:
                wrapper = getattr(layer.self_attn, name, None)
                if wrapper is not None and hasattr(wrapper, "cera"):
                    wrapper.cera.track_activation = True
                    tracked.append(wrapper.cera)

        model.eval()
        with torch.no_grad():
            model(inputs.to(DEVICE))

        er_vals = [
            compute_effective_rank(mod.last_delta)
            for mod in tracked if mod.last_delta is not None
        ]
        avg_er = float(np.mean(er_vals)) if er_vals else 0.0
        print(f"{spec.label},{avg_er:.4f}")

        del model, ckpt
        torch.cuda.empty_cache()
        gc.collect()


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    args = parse_args()

    target_layers = [int(x.strip()) for x in args.target_layers.split(",")]
    base_model    = args.base_model

    print(
        f"[INFO] Loading {args.num_samples} MathInstruct samples for activation profiling..."
    )
    inputs = load_profiling_inputs(base_model, args.num_samples)

    if args.mode == "manifold":
        rank    = args.rank    if args.rank    is not None else 64
        dropout = args.dropout if args.dropout is not None else 0.1
        print(f"=== Manifold Expansion: CeRA R{rank} vs LoRA R{rank} ===", flush=True)
        configs: ConfigList = [
            (f"LoRA (R={rank})", args.lora_path, "LoRA", rank, 0.0),
            (f"CeRA (R={rank})", args.cera_path, "CeRA", rank, dropout),
        ]
        _run_analysis(configs, inputs, base_model, target_layers)

    elif args.mode == "lr_sensitivity":
        rank    = args.rank    if args.rank    is not None else 16
        dropout = args.dropout if args.dropout is not None else 0.1
        print(f"=== LR Sensitivity: LoRA R{rank} vs CeRA R{rank} (Low/High LR) ===", flush=True)
        configs = [
            (f"LoRA (R={rank})",          args.lora_path,      "LoRA", rank, 0.0),
            (f"CeRA (R={rank}, Low LR)",  args.cera_low_path,  "CeRA", rank, dropout),
            (f"CeRA (R={rank}, High LR)", args.cera_high_path, "CeRA", rank, dropout),
        ]
        _run_analysis(configs, inputs, base_model, target_layers)

    elif args.mode == "dropout":
        print("=== Plasticity Control: CeRA Dropout Study ===", flush=True)
        rank = args.rank if args.rank is not None else 512
        configs = [
            (f"CeRA (R={rank}, D=0)",   args.no_dropout_path, "CeRA", rank, 0.0),
            (f"CeRA (R={rank}, D=0.3)", args.dropout_path,    "CeRA", rank, 0.3),
        ]
        _run_analysis(configs, inputs, base_model, target_layers)

    elif args.mode == "ablation":
        rank = args.rank if args.rank is not None else 128
        print(f"=== Ablation ER Analysis (R={rank}) ===", flush=True)
        specs: List[AblationSpec] = [
            AblationSpec("CeRA (Full)",      args.full_path,      "silu",     ["q_proj", "v_proj"], 0.3),
            AblationSpec("(a) Granularity",  args.module_path,    "silu",     ["o_proj"],            0.3),
            AblationSpec("(b) Identity",     args.linear_path,    "identity", ["q_proj", "v_proj"], 0.3),
            AblationSpec("(b) ReLU",         args.relu_path,      "relu",     ["q_proj", "v_proj"], 0.3),
            AblationSpec("(c) No Dropout",   args.nodropout_path, "silu",     ["q_proj", "v_proj"], 0.0),
        ]
        _run_ablation_analysis(specs, inputs, rank, base_model, target_layers)


if __name__ == "__main__":
    main()
