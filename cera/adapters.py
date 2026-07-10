"""
CeRA and LoRA adapter definitions -- single source of truth for the whole repo.

CeRA (Capacity-enhanced Rank Adaptation) injects a non-linear parallel branch
into each target projection:

    output = W_0 x  +  B( Dropout( act_fn( A x ) ) )

This breaks LoRA's "linear ceiling": because of the non-linearity, the adapter
can express functions outside the column span of W_0, giving access to the full
output space rather than only a low-dimensional subspace.

Toggle between CeRA and LoRA at injection time via apply_cera() / apply_lora().
Both wrappers expose a track_activation flag for downstream SVD/ER analysis.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------------------
# CeRA
# ------------------------------------------------------------------------------

class CeRAAdapter(nn.Module):
    """
    Two-layer MLP adapter with SiLU gating and structural Dropout.

    Architecture (matching Eq. (2) of the paper, delta = B D(sigma(A x))):
        x  ->  A  ->  act_fn  ->  Dropout  ->  B  ->  delta

    A and B follow the LoRA naming convention: A projects the input down to
    the latent dimension r, B projects back to the output dimension.

    Hidden (latent) dimension is set by expansion_factor:
        hidden_dim = int(input_dim * expansion_factor)

    For rank r on a d-dimensional projection (e.g. q_proj in Llama-3-8B, d=4096):
        expansion_factor = r / d

    Initialization:
        A : Kaiming normal  (safe for SiLU / ReLU)
        B : zeros           (adapter contributes nothing at step 0)

    Checkpoints saved before the 2026-07 A/B rename use the legacy key names
    up_proj/down_proj; _load_from_state_dict remaps them.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        expansion_factor: float = 0.5,
        dropout: float = 0.3,
        act_fn: str = "silu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        hidden_dim = int(input_dim * expansion_factor)

        self.A = nn.Linear(input_dim, hidden_dim, bias=False, device=device, dtype=dtype)

        if act_fn == "silu":
            self.act_fn = nn.SiLU()
        elif act_fn == "relu":
            self.act_fn = nn.ReLU()
        elif act_fn == "identity":
            self.act_fn = nn.Identity()
        else:
            raise ValueError(
                f"Unknown activation {act_fn!r}. Choose from 'silu', 'relu', 'identity'."
            )

        self.B = nn.Linear(hidden_dim, output_dim, bias=False, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

        # Activation tracking -- enable before a forward pass to capture last_delta
        # for SVD / Effective Rank analysis. Disabled by default (no overhead).
        self.track_activation: bool = False
        self.last_delta: Optional[torch.Tensor] = None

        nn.init.kaiming_normal_(self.A.weight, nonlinearity="relu")
        nn.init.zeros_(self.B.weight)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Backward compatibility: checkpoints saved before the A/B rename
        # (2026-07) store the projections as up_proj/down_proj.
        # Remap the legacy keys in place. Without this remap, strict=False
        # loading would silently skip both matrices and (because B is
        # zero-initialized) evaluate the frozen base model.
        for old_name, new_name in (("up_proj", "A"), ("down_proj", "B")):
            old_key = f"{prefix}{old_name}.weight"
            if old_key in state_dict:
                state_dict[f"{prefix}{new_name}.weight"] = state_dict.pop(old_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.B(self.dropout(self.act_fn(self.A(x))))
        if self.track_activation:
            self.last_delta = delta.detach().cpu().float()
        return delta


class CeRAWrapper(nn.Module):
    """
    Wraps an existing Linear layer with a parallel CeRA adapter.

        forward(x) = original_layer(x) + CeRAAdapter(x)

    The original layer's parameters are frozen; only the CeRAAdapter trains.
    Device and dtype are inherited from the original layer automatically.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        input_dim: int,
        output_dim: int,
        expansion_factor: float,
        dropout: float = 0.3,
        act_fn: str = "silu",
    ):
        super().__init__()
        self.original_layer = original_layer
        for p in original_layer.parameters():
            p.requires_grad = False  # freeze pre-trained weights

        self.cera = CeRAAdapter(
            input_dim,
            output_dim,
            expansion_factor,
            dropout=dropout,
            act_fn=act_fn,
            device=original_layer.weight.device,
            dtype=original_layer.weight.dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original_layer(x) + self.cera(x)


# ------------------------------------------------------------------------------
# LoRA  (baseline)
# ------------------------------------------------------------------------------

class LoRAWrapper(nn.Module):
    """
    Standard LoRA adapter: low-rank decomposition delta = (x A B) * (alpha / rank).

    Initialization:
        lora_A : small Gaussian noise  (breaks symmetry)
        lora_B : zeros                 (adapter contributes nothing at step 0)

    The effective rank of the output subspace is bounded by `rank`.
    Unlike CeRA, this is a linear operation, limiting the representable
    function space to a low-dimensional sub-manifold of the output space.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original_layer = original_layer
        for p in original_layer.parameters():
            p.requires_grad = False

        dev, dt = original_layer.weight.device, original_layer.weight.dtype
        self.lora_A = nn.Parameter(torch.randn(in_dim, rank, device=dev, dtype=dt) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim, device=dev, dtype=dt))
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

        self.track_activation: bool = False
        self.last_delta: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        if self.track_activation:
            self.last_delta = delta.detach().cpu().float()
        return self.original_layer(x) + delta


# ------------------------------------------------------------------------------
# DoRA  (baseline)
# ------------------------------------------------------------------------------

class DoRAWrapper(nn.Module):
    """
    DoRA (Weight-Decomposed Low-Rank Adaptation, Liu et al. 2024).

    Decomposes the adapted weight into a learnable magnitude vector m and a
    unit-norm direction matrix, then applies a LoRA delta to the direction:

        W_dora = m * (W_0 + B A * scale) / ||(W_0 + B A * scale)||_col

    where ||.||_col denotes the L2 norm of each row (per-output-neuron norm).

    Trainable parameters:
        m      -- magnitude vector, shape (out_dim,), init = ||W_0||_col
        lora_A -- shape (in_dim, rank),  Gaussian init
        lora_B -- shape (rank, out_dim), zero init

    This allows larger directional updates than plain LoRA without blowing up
    the weight norm, and consistently outperforms LoRA at matched rank.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: int = 32,
    ):
        super().__init__()
        self.original_layer = original_layer
        for p in original_layer.parameters():
            p.requires_grad = False

        dev, dt = original_layer.weight.device, original_layer.weight.dtype

        # LoRA matrices (same convention as LoRAWrapper)
        self.lora_A = nn.Parameter(torch.randn(in_dim, rank, device=dev, dtype=dt) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim, device=dev, dtype=dt))
        self.scaling = alpha / rank

        # Magnitude: initialized to per-row L2 norm of W_0
        # W_0 shape: (out_dim, in_dim) -- norm of each row = norm of each output neuron
        with torch.no_grad():
            row_norms = original_layer.weight.norm(dim=1)  # (out_dim,)
        self.m = nn.Parameter(row_norms.clone().to(dtype=dt))

        self.track_activation: bool = False
        self.last_delta: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_0 = self.original_layer.weight  # (out_dim, in_dim), frozen

        # LoRA weight delta in weight space: (out_dim, rank) @ (rank, in_dim)
        delta_W = (self.lora_B.T @ self.lora_A.T) * self.scaling  # (out_dim, in_dim)

        W_adapted = W_0 + delta_W  # (out_dim, in_dim)

        # Per-row L2 norm (clamp to avoid division by zero)
        row_norms = W_adapted.norm(dim=1, keepdim=True).clamp(min=1e-8)  # (out_dim, 1)

        # Weight-decomposed weight: scale direction by learnable magnitude
        W_dora = (self.m.unsqueeze(1) / row_norms) * W_adapted  # (out_dim, in_dim)

        out = F.linear(x, W_dora, self.original_layer.bias)

        if self.track_activation:
            base = self.original_layer(x)
            self.last_delta = (out - base).detach().cpu().float()

        return out


# ------------------------------------------------------------------------------
# Injection helpers
# ------------------------------------------------------------------------------

def apply_cera(
    model: nn.Module,
    expansion_factor: float,
    dropout: float = 0.3,
    act_fn: str = "silu",
    target_modules: Optional[List[str]] = None,
    layer_indices: Optional[List[int]] = None,
) -> nn.Module:
    """
    Inject CeRA adapters into self-attention projections of a Llama model.

    Args:
        model:            HuggingFace CausalLM (e.g. Llama-3-8B).
        expansion_factor: hidden_dim / input_dim ratio.
                          For rank r on attention dim d: r / d.
        dropout:          Structural dropout rate (default 0.3).
        act_fn:           'silu' | 'relu' | 'identity'.
        target_modules:   Projection names to adapt. Default: ['q_proj', 'v_proj'].
        layer_indices:    Transformer layers to inject into. None = all layers.

    Returns:
        Model with CeRA adapters injected (in-place modification).
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    layer_tag = "all" if layer_indices is None else layer_indices
    print(
        f"[INFO] Applying CeRA | Exp={expansion_factor:.4f} | "
        f"Dropout={dropout} | Act={act_fn} | "
        f"Targets={target_modules} | Layers={layer_tag}"
    )

    for layer_idx, layer in enumerate(model.model.layers):
        if layer_indices is not None and layer_idx not in layer_indices:
            continue
        for name, module in layer.self_attn.named_children():
            if name in target_modules:
                wrapper = CeRAWrapper(
                    module,
                    module.in_features,
                    module.out_features,
                    expansion_factor,
                    dropout=dropout,
                    act_fn=act_fn,
                )
                setattr(layer.self_attn, name, wrapper)

    return model


def apply_lora(
    model: nn.Module,
    rank: int,
    alpha: int = 32,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    layer_indices: Optional[List[int]] = None,
) -> nn.Module:
    """
    Inject standard LoRA adapters into self-attention projections.

    Args:
        model:          HuggingFace CausalLM.
        rank:           Low-rank bottleneck dimension.
        alpha:          Scaling factor; effective scale = alpha / rank.
        dropout:        Dropout applied to input x before the LoRA branch (default 0.0).
        target_modules: Projection names. Default: ['q_proj', 'v_proj'].
        layer_indices:  Layers to inject. None = all layers.

    Returns:
        Model with LoRA adapters injected (in-place modification).
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    layer_tag = "all" if layer_indices is None else layer_indices
    print(
        f"[INFO] Applying LoRA | Rank={rank} | Alpha={alpha} | Dropout={dropout} | "
        f"Targets={target_modules} | Layers={layer_tag}"
    )

    for layer_idx, layer in enumerate(model.model.layers):
        if layer_indices is not None and layer_idx not in layer_indices:
            continue
        for name, module in layer.self_attn.named_children():
            if name in target_modules:
                wrapper = LoRAWrapper(
                    module,
                    module.in_features,
                    module.out_features,
                    rank,
                    alpha,
                    dropout,
                )
                setattr(layer.self_attn, name, wrapper)

    return model


def apply_dora(
    model: nn.Module,
    rank: int,
    alpha: int = 32,
    target_modules: Optional[List[str]] = None,
    layer_indices: Optional[List[int]] = None,
) -> nn.Module:
    """
    Inject DoRA adapters into self-attention projections.

    Args:
        model:          HuggingFace CausalLM.
        rank:           Low-rank bottleneck dimension.
        alpha:          Scaling factor; effective scale = alpha / rank.
        target_modules: Projection names. Default: ['q_proj', 'v_proj'].
        layer_indices:  Layers to inject. None = all layers.

    Returns:
        Model with DoRA adapters injected (in-place modification).
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    layer_tag = "all" if layer_indices is None else layer_indices
    print(
        f"[INFO] Applying DoRA | Rank={rank} | Alpha={alpha} | "
        f"Targets={target_modules} | Layers={layer_tag}"
    )

    for layer_idx, layer in enumerate(model.model.layers):
        if layer_indices is not None and layer_idx not in layer_indices:
            continue
        for name, module in layer.self_attn.named_children():
            if name in target_modules:
                wrapper = DoRAWrapper(
                    module,
                    module.in_features,
                    module.out_features,
                    rank,
                    alpha,
                )
                setattr(layer.self_attn, name, wrapper)

    return model
