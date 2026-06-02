"""
SVD-based analysis utilities for CeRA / LoRA adapter output inspection.

Key functions
-------------
  compute_effective_rank(matrix)
      Shannon-entropy effective rank of adapter activations.

  compute_manifold_dim(matrix, threshold)
      Minimum singular values to explain threshold fraction of variance.

  get_singular_values(ckpt_path, inputs, config, ...)
      Load a checkpoint, run a forward pass, and return the averaged,
      normalised singular value spectrum across target layers.

  find_best_checkpoint(ckpt_dir, model_type)
      Return the .pt file with the best validation checkpoint.

  scan_experiment_dirs(base_dir, dataset, model_types)
      Walk a results/ directory and collect matching experiment folders.
"""

import gc
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


# ------------------------------------------------------------------------------
# Effective Rank
# ------------------------------------------------------------------------------

def compute_effective_rank(matrix: torch.Tensor) -> float:
    """
    Effective Rank (ER) of an activation matrix via Shannon entropy.

        ER = exp( -sum_i( p_i * log(p_i) ) )   where  p_i = sigma_i / sum_j(sigma_j)

    A high ER means the adapter uses a broad, distributed subspace;
    a low ER means the adapter output collapses into a few directions.

    Reference: Roy & Vetterli (2007), "The effective rank: A measure of
    effective dimensionality." EUSIPCO.

    Args:
        matrix: Activation tensor [N, D] or [B, T, D] (auto-flattened).

    Returns:
        Scalar effective rank.
    """
    if matrix.dim() > 2:
        matrix = matrix.view(-1, matrix.size(-1))

    # Centre activations (PCA-style)
    matrix = matrix - matrix.mean(dim=0)

    try:
        _, S, _ = torch.svd(matrix.float())
        S_norm  = S / (S.sum() + 1e-12)
        entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-10))
        return torch.exp(entropy).item()
    except RuntimeError:
        return 0.0


# ------------------------------------------------------------------------------
# Manifold Dimensionality
# ------------------------------------------------------------------------------

def compute_manifold_dim(matrix: torch.Tensor, threshold: float = 0.90) -> int:
    """
    Manifold Dimensionality: the minimum number of singular values (principal
    components) needed to explain `threshold` fraction of total variance in the
    adapter's output activations.

    Formally, the smallest k such that:
        sum_{i=1}^{k} sigma_i^2  >=  threshold * sum_j sigma_j^2

    Args:
        matrix:    Activation tensor [N, D] or [B, T, D] (auto-flattened).
        threshold: Explained-variance fraction (default 0.90 -> 90%).

    Returns:
        Integer k >= 1.
    """
    if matrix.dim() > 2:
        matrix = matrix.view(-1, matrix.size(-1))

    matrix = matrix - matrix.mean(dim=0)  # centre

    try:
        _, S, _ = torch.svd(matrix.float())
    except RuntimeError:
        return 0

    S       = S.abs()
    var     = S ** 2
    cumvar  = torch.cumsum(var, dim=0)
    total   = var.sum()

    if total < 1e-12:
        return 0

    k = int((cumvar / total < threshold).sum().item()) + 1
    return min(k, len(S))


# ------------------------------------------------------------------------------
# SVD spectrum computation
# ------------------------------------------------------------------------------

def _enable_tracking(
    model: nn.Module,
    model_type: str,
    layer_indices: List[int],
    target_modules: List[str],
) -> List[nn.Module]:
    """Enable track_activation on adapter sub-modules and return the list."""
    tracked = []
    for layer_idx in layer_indices:
        layer = model.model.layers[layer_idx]
        for name in target_modules:
            wrapper = getattr(layer.self_attn, name, None)
            if wrapper is None:
                continue
            if model_type == "CeRA" and hasattr(wrapper, "cera"):
                wrapper.cera.track_activation = True
                tracked.append(wrapper.cera)
            elif model_type == "LoRA" and hasattr(wrapper, "lora_A"):
                wrapper.track_activation = True
                tracked.append(wrapper)
    return tracked


def get_singular_values(
    ckpt_path: str,
    inputs: torch.Tensor,
    config: Dict,
    model_name: str,
    hf_token: Optional[str],
    layer_indices: List[int],
    device: str = "cuda",
    dtype=torch.bfloat16,
) -> Optional[List[float]]:
    """
    Load a checkpoint, run a single forward pass, and compute the averaged
    normalised singular value spectrum of adapter output activations across
    the specified transformer layers.

    The returned spectrum is averaged over all tracked (layer * projection)
    combinations so that it can be compared across adapter types.

    Args:
        ckpt_path:     Path to a .pt checkpoint (produced by save_checkpoint).
        inputs:        Tokenised profiling tensor [N, seq_len].
        config:        Experiment config dict.  Required keys:
                       'type' ('CeRA'|'LoRA'), 'rank', 'dropout', 'act_fn',
                       'target_modules'.
        model_name:    HuggingFace model identifier.
        hf_token:      HuggingFace access token (for gated models).
        layer_indices: Transformer layer indices to analyse.
        device:        Compute device.
        dtype:         Model dtype (bfloat16 recommended for Llama-3-8B).

    Returns:
        List of averaged, normalised singular values, or None on failure.
    """
    from cera.adapters import apply_cera, apply_lora

    model_type     = config.get("type", "CeRA")
    rank           = config.get("rank", 64)
    dropout        = config.get("dropout", 0.0)
    act_fn         = config.get("act_fn", "silu")
    tm_raw         = config.get("target_modules", "q_proj,v_proj")
    target_modules = [t.strip() for t in tm_raw.split(",")] if isinstance(tm_raw, str) else tm_raw

    print(
        f"[SVD] {model_type} R={rank} D={dropout} Act={act_fn} | "
        f"{os.path.basename(ckpt_path)}"
    )

    # 1. Load clean base model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, token=hf_token, torch_dtype=dtype, device_map=device
        )
    except Exception as exc:
        print(f"[ERROR] Base model load failed: {exc}")
        return None

    for p in model.parameters():
        p.requires_grad = False

    # 2. Inject adapter structure (atten_dim auto-detected from model config)
    atten_dim = model.config.hidden_size
    if model_type == "CeRA":
        exp = rank / atten_dim
        model = apply_cera(
            model, exp, dropout=dropout, act_fn=act_fn,
            target_modules=target_modules, layer_indices=layer_indices,
        )
    elif model_type == "LoRA":
        model = apply_lora(
            model, rank, target_modules=target_modules, layer_indices=layer_indices
        )
    else:
        print(f"[ERROR] Unknown model type: {model_type!r}")
        return None

    model.to(device=device, dtype=dtype)

    # 3. Load checkpoint weights
    if not os.path.exists(ckpt_path):
        print(f"[WARN] Checkpoint not found: {ckpt_path}")
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[SVD] Checkpoint loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    except Exception as exc:
        print(f"[ERROR] Checkpoint load failed: {exc}")
        return None

    # 4. Enable activation tracking
    tracked = _enable_tracking(model, model_type, layer_indices, target_modules)
    if not tracked:
        print(f"[WARN] No trackable modules found in layers {layer_indices}.")
        return None

    # 5. Single forward pass (eval mode -- dropout is inactive)
    model.eval()
    with torch.no_grad():
        try:
            model(inputs.to(device))
        except Exception as exc:
            print(f"[ERROR] Forward pass failed: {exc}")
            return None

    # 6. Average SVD across all tracked modules
    accumulated_S: Optional[np.ndarray] = None
    count = 0

    for mod in tracked:
        if mod.last_delta is None:
            continue

        mat = mod.last_delta
        if mat.dim() > 2:
            mat = mat.view(-1, mat.size(-1))
        mat = mat - mat.mean(dim=0)  # centre

        try:
            _, S, _ = torch.svd(mat.float())
            if S.sum() > 0:
                S = S / S.sum()
            S_np = S.cpu().numpy()

            if accumulated_S is None:
                accumulated_S = S_np
            else:
                n = min(len(accumulated_S), len(S_np))
                accumulated_S = accumulated_S[:n] + S_np[:n]
            count += 1
        except Exception as exc:
            print(f"[WARN] SVD failed on one module: {exc}")

    del model, ckpt
    torch.cuda.empty_cache()
    gc.collect()

    if count == 0 or accumulated_S is None:
        return None

    return (accumulated_S / count).tolist()


# ------------------------------------------------------------------------------
# Experiment directory utilities
# ------------------------------------------------------------------------------

def find_best_checkpoint(ckpt_dir: Path, model_type: str) -> Optional[Path]:
    """
    Return the best saved checkpoint from ckpt_dir.

    Priority:
      1. *_best_*.pt  (new format: explicitly flagged best validation checkpoint)
      2. *_ckpt_<N>.pt  (old format: fallback to highest data-seen count)
    """
    prefix = model_type.lower()

    # New format: cera_ckpt_best_<step>.pt
    best_files = sorted(ckpt_dir.glob(f"{prefix}_ckpt_best_*.pt"))
    if best_files:
        return best_files[-1]

    # Old format: cera_ckpt_<data_count>.pt -- pick highest count
    pat = re.compile(rf"{prefix}_ckpt_(\d+)\.pt$")
    best, best_n = None, -1
    for f in ckpt_dir.iterdir():
        m = pat.match(f.name)
        if m:
            n = int(m.group(1))
            if n > best_n:
                best_n, best = n, f
    return best


def scan_experiment_dirs(
    base_dir: str,
    dataset: str,
    model_types: Optional[List[str]] = None,
) -> List[Tuple[Path, str]]:
    """
    Walk a results/ directory and collect matching experiment folders.

    Folder naming convention (set by train.py):
        Exp_{model_type}_{dataset}_R{rank}_lr{lr}_{act}_{targets}_D{drop}_E{ep}_{ts}

    Args:
        base_dir:    Root directory to scan (typically 'results').
        dataset:     Dataset tag to filter on ('math' | 'code' | 'orca').
        model_types: Adapter types to include. Default: ['CeRA', 'LoRA'].

    Returns:
        List of (folder_path, model_type_key) tuples.
    """
    if model_types is None:
        model_types = ["CeRA", "LoRA"]

    found: List[Tuple[Path, str]] = []
    for root, dirs, _ in os.walk(base_dir):
        for d in dirs:
            for mt in model_types:
                if d.startswith(f"Exp_{mt}_{dataset}"):
                    found.append((Path(root) / d, mt))

    print(f"[SCAN] {len(found)} experiment folders found in '{base_dir}'.")
    return found
