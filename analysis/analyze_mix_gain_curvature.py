#!/usr/bin/env python3
"""Split the learned-mix correction into gain, offset, and curvature on NLL.

The learned-mix correction of one adapter is c(z) = gamma * (SiLU(z) - z).
On a calibration split taken from the training rows we fit, per module,

    c(z) ~= C z + b

at three levels (scalar C = s*I, diagonal C, full r x r C).  On a held-out
split we then swap the correction of the *same checkpoint* for each piece and
measure teacher-forced completion NLL and KL to the unmodified model:

    full              z + c(z)
    zero              z
    gain_scalar       z + s z
    offset_scalar     z + b
    affine_scalar     z + s z + b
    curvature_scalar  z + c(z) - s z - b
    affine_diag / curvature_diag / affine_matrix / curvature_matrix

Comparisons use a problem-cluster bootstrap over held-out samples.
"""

import argparse
import json
import os
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate
from analyze_mix_correction import encode_batch
from cera.adapters import CeRAWrapper
from cera.data import METAMATH_TEST_OFF, METAMATH_TRAIN_N

FIT_LEVELS = ("scalar", "diag", "matrix")
CONDITIONS = (
    "full", "zero",
    "gain_scalar", "offset_scalar", "affine_scalar", "curvature_scalar",
    "affine_diag", "curvature_diag",
    "affine_matrix", "curvature_matrix",
)
PROJECTIONS = ("q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gain/offset/curvature interventions on a learned-mix CeRA checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_calibration", type=int, default=200,
                        help="Training rows used to fit the linear pieces.")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Held-out rows used for NLL / KL.")
    parser.add_argument("--eval_offset", type=int, default=METAMATH_TEST_OFF)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--fit_scope", choices=("all", "response"), default="all",
                        help="Token positions used to fit and describe the pieces. "
                             "Interventions always apply to every position.")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32",
                        help="Base-model dtype. bfloat16 matches evaluate.py but rounds "
                             "base + delta to ~0.4%%, which is coarse next to a ~1%% "
                             "change of the adapter delta.")
    parser.add_argument("--ridge", type=float, default=1e-6,
                        help="Relative ridge for the full-matrix fit.")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261007)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Moments and fits
# ---------------------------------------------------------------------------

class Moments:
    """Sufficient statistics of (z, c) over selected token positions."""

    def __init__(self, rank):
        self.n = 0
        self.sum_z = torch.zeros(rank, dtype=torch.float64)
        self.sum_c = torch.zeros(rank, dtype=torch.float64)
        self.zz = torch.zeros(rank, rank, dtype=torch.float64)
        self.cz = torch.zeros(rank, rank, dtype=torch.float64)
        self.cc = 0.0

    def update(self, z, c):
        z = z.double()
        c = c.double()
        self.n += z.shape[0]
        self.sum_z += z.sum(dim=0).cpu()
        self.sum_c += c.sum(dim=0).cpu()
        self.zz += (z.T @ z).cpu()
        self.cz += (c.T @ z).cpu()
        self.cc += float(c.square().sum())


def fit_linear_pieces(moments, ridge=1e-6):
    """Least-squares c ~= C z + b for scalar, diagonal, and full C."""
    if moments.n < 2:
        raise ValueError("Not enough calibration tokens to fit the correction.")
    rank = moments.sum_z.numel()
    mean_z = moments.sum_z / moments.n
    mean_c = moments.sum_c / moments.n
    cov_zz = moments.zz / moments.n - torch.outer(mean_z, mean_z)
    cov_cz = moments.cz / moments.n - torch.outer(mean_c, mean_z)
    eye = torch.eye(rank, dtype=torch.float64)

    scalar = float(torch.trace(cov_cz) / torch.trace(cov_zz))
    diag = torch.diagonal(cov_cz) / torch.diagonal(cov_zz).clamp_min(1e-30)
    damping = ridge * float(torch.trace(cov_zz)) / rank
    matrix = torch.linalg.solve(cov_zz + damping * eye, cov_cz.T).T

    fits = {}
    for level, C in (("scalar", scalar * eye), ("diag", torch.diag(diag)), ("matrix", matrix)):
        fits[level] = {"C": C, "b": mean_c - C @ mean_z}
    # Uncentered no-offset gain, i.e. s* = sum <z, c> / sum ||z||^2.
    fits["uncentered_scalar"] = float(torch.trace(moments.cz) / torch.trace(moments.zz))
    return fits


def piece_energies(moments, C, b):
    """Energies of C z, b, and c - C z - b, all summed over tokens."""
    gain = float(((C @ moments.zz) * C).sum())
    offset = moments.n * float(b @ b)
    residual = (
        moments.cc
        - 2.0 * float((C * moments.cz).sum())
        - 2.0 * float(b @ moments.sum_c)
        + gain
        + 2.0 * float(b @ (C @ moments.sum_z))
        + offset
    )
    return {"gain": gain, "offset": offset, "residual": max(residual, 0.0)}


def substitute_correction(condition, z, gamma, fit):
    """Return the term added to z under one intervention."""
    if condition == "zero":
        return torch.zeros_like(z)
    correction = gamma * (F.silu(z) - z)
    if condition == "full":
        return correction
    kind, level = condition.split("_")
    C, b = fit[level]["C"], fit[level]["b"]
    if kind == "offset":
        return b.expand_as(z)
    if kind == "gain":
        return z @ C.T
    linear = z @ C.T + b
    if kind == "affine":
        return linear
    if kind == "curvature":
        return correction - linear
    raise ValueError(f"Unknown condition {condition!r}.")


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

class InterventionController:
    """Recomputes each learned-mix adapter output under the active condition."""

    def __init__(self, model):
        self.condition = "full"
        self.collect = None      # dict name -> Moments, or None
        self.mask = None
        self.enabled = True
        self.adapters = {}
        self.fits = {}
        self.handles = []
        for name, module in model.named_modules():
            if not isinstance(module, CeRAWrapper):
                continue
            adapter = module.cera
            if (adapter.mix_mode != "learned_mix" or adapter.variant != "peft_aligned"
                    or adapter.recurrent_steps):
                raise ValueError(f"{name} is not a non-recurrent learned-mix adapter.")
            self.adapters[name] = adapter
            self.handles.append(adapter.register_forward_hook(self._hook(name, adapter)))
        if not self.adapters:
            raise ValueError("No CeRA adapters found in model.")

    def rank(self, name):
        return self.adapters[name].A.out_features

    def new_moments(self):
        return {name: Moments(self.rank(name)) for name in self.adapters}

    def set_fits(self, fits):
        self.fits = {}
        for name, adapter in self.adapters.items():
            weight = adapter.A.weight
            self.fits[name] = {
                level: {
                    key: fits[name][level][key].to(device=weight.device, dtype=weight.dtype)
                    for key in ("C", "b")
                }
                for level in FIT_LEVELS
            }

    def _hook(self, name, adapter):
        def replace(_module, inputs, _output):
            if not self.enabled:
                return None
            z = adapter.A(adapter.dropout(inputs[0].to(adapter.A.weight.dtype)))
            if self.collect is not None:
                selected = z[self.mask.to(z.device)]
                if selected.numel():
                    self.collect[name].update(
                        selected, adapter.gamma * (F.silu(selected) - selected)
                    )
            added = substitute_correction(
                self.condition, z, adapter.gamma, self.fits.get(name)
            )
            return adapter.B(z + added) * adapter.scaling
        return replace

    def close(self):
        for handle in self.handles:
            handle.remove()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def target_log_probs(logits, input_ids, token_mask):
    """Log-probabilities at positions predicting the scored response tokens."""
    target_mask = token_mask[:, 1:].to(logits.device)
    log_probs = F.log_softmax(logits[:, :-1][target_mask].float(), dim=-1)
    targets = input_ids[:, 1:][target_mask]
    rows = target_mask.nonzero()[:, 0]
    return log_probs, targets, rows


def bootstrap_delta(delta_sums, token_counts, draws, rng):
    """Token-weighted mean of a per-sample sum, resampling whole problems."""
    delta_sums = np.asarray(delta_sums, dtype=np.float64)
    token_counts = np.asarray(token_counts, dtype=np.float64)
    estimate = delta_sums.sum() / token_counts.sum()
    index = rng.integers(0, len(delta_sums), size=(draws, len(delta_sums)))
    samples = delta_sums[index].sum(axis=1) / token_counts[index].sum(axis=1)
    tail = min((samples <= 0).mean(), (samples >= 0).mean())
    return {
        "delta": float(estimate),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "p_two_sided": float(min(1.0, 2.0 * tail)),
        "fraction_samples_lower": float((delta_sums < 0).mean()),
    }


def describe_modules(controller, fits, calibration, heldout):
    rows = []
    for name, adapter in controller.adapters.items():
        layer = int(name.split(".layers.", 1)[1].split(".", 1)[0])
        row = {
            "module": name,
            "layer": layer,
            "layer_band": "early" if layer <= 7 else "middle" if layer <= 23 else "late",
            "projection": next(p for p in PROJECTIONS if f".{p}" in name),
            "gamma": float(adapter.gamma.detach().cpu()),
            "scalar_gain": float(fits[name]["scalar"]["C"][0, 0]),
            "uncentered_scalar_gain": fits[name]["uncentered_scalar"],
            "offset_norm": float(fits[name]["scalar"]["b"].norm()),
        }
        for split, moments in (("calibration", calibration[name]), ("heldout", heldout[name])):
            z_energy = float(torch.trace(moments.zz))
            c_energy = moments.cc or float("nan")
            row[f"{split}_tokens"] = moments.n
            row[f"{split}_correction_ratio"] = (moments.cc / z_energy) ** 0.5
            for level in FIT_LEVELS:
                energy = piece_energies(moments, fits[name][level]["C"], fits[name][level]["b"])
                row[f"{split}_r2_{level}"] = 1.0 - energy["residual"] / c_energy
                row[f"{split}_curvature_ratio_{level}"] = (energy["residual"] / z_energy) ** 0.5
                if level == "scalar":
                    row[f"{split}_gain_ratio"] = (energy["gain"] / z_energy) ** 0.5
                    row[f"{split}_offset_ratio"] = (energy["offset"] / z_energy) ** 0.5
            s0 = fits[name]["uncentered_scalar"]
            eye = torch.eye(moments.sum_z.numel(), dtype=torch.float64)
            energy = piece_energies(moments, s0 * eye, torch.zeros_like(moments.sum_z))
            row[f"{split}_r2_uncentered_scalar"] = 1.0 - energy["residual"] / c_energy
        rows.append(row)
    return rows


def group_means(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    numeric = [k for k, v in rows[0].items() if isinstance(v, float)]
    return {
        str(group): {"count": len(members),
                     **{k: sum(m[k] for m in members) / len(members) for k in numeric}}
        for group, members in groups.items()
    }


def main():
    args = parse_args()
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}. Choose from {CONDITIONS}.")
    conditions = ["full", "zero"] + [c for c in conditions if c not in ("full", "zero")]
    if min(args.num_calibration, args.num_samples, args.batch_size, args.bootstrap) < 1:
        raise ValueError("Sample counts, batch_size, and bootstrap must be positive.")
    if args.num_calibration > METAMATH_TRAIN_N:
        raise ValueError("Calibration rows must come from the training split.")
    if args.eval_offset < METAMATH_TRAIN_N:
        raise ValueError("--eval_offset overlaps the training split.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint.get("config") or {}
    if config.get("cera_mix_mode") != "learned_mix":
        raise ValueError("Checkpoint is not learned-mix CeRA.")
    if config.get("dataset", "metamathqa") != "metamathqa":
        raise ValueError("Splits here assume a MetaMathQA-trained checkpoint.")
    del checkpoint

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN is not set.")
    evaluate.DTYPE = getattr(torch, args.dtype)
    namespace = Namespace(
        base_model=config["model"], adapter_type="cera", rank=config["rank"],
        checkpoint=str(args.checkpoint), adapter_format="legacy",
        alpha=config["alpha"], act_fn=config["act_fn"], dropout=config["dropout"],
        target_modules=config["target_modules"], explicit_options=set(),
    )
    model, tokenizer = evaluate.load_model_with_adapter(namespace, hf_token)
    tokenizer.padding_side = "right"
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for response masking.")

    dataset = load_dataset("meta-math/MetaMathQA", split="train")
    calibration_rows = list(dataset.select(range(args.num_calibration)))
    eval_stop = min(args.eval_offset + args.num_samples, len(dataset))
    eval_rows = list(dataset.select(range(args.eval_offset, eval_stop)))
    device = model.get_input_embeddings().weight.device
    controller = InterventionController(model)

    def forward(rows, scope):
        encoded, response_mask = encode_batch(tokenizer, rows, args.max_length, "response")
        controller.mask = (
            response_mask if scope == "response" else encoded["attention_mask"].bool()
        )
        inputs = {key: value.to(device) for key, value in encoded.items()}
        return inputs, response_mask

    try:
        # Pass 1: fit the linear pieces on training rows.
        calibration = controller.new_moments()
        controller.condition, controller.collect = "full", calibration
        for start in tqdm(range(0, len(calibration_rows), args.batch_size), desc="Calibrating"):
            inputs, _ = forward(calibration_rows[start:start + args.batch_size], args.fit_scope)
            with torch.inference_mode():
                model(**inputs, use_cache=False)
        controller.collect = None
        fits = {name: fit_linear_pieces(calibration[name], args.ridge) for name in calibration}
        controller.set_fits(fits)

        # Pass 2: held-out NLL / KL under every condition.
        heldout = controller.new_moments()
        nll = {c: np.zeros(len(eval_rows)) for c in conditions}
        kl = {c: np.zeros(len(eval_rows)) for c in conditions}
        token_counts = np.zeros(len(eval_rows))
        hook_parity = None
        for start in tqdm(range(0, len(eval_rows), args.batch_size), desc="Scoring"):
            inputs, response_mask = forward(eval_rows[start:start + args.batch_size],
                                            args.fit_scope)
            full_log_probs = None
            for condition in conditions:
                controller.condition = condition
                controller.collect = heldout if condition == "full" else None
                with torch.inference_mode():
                    logits = model(**inputs, use_cache=False).logits
                    log_probs, targets, rows = target_log_probs(
                        logits, inputs["input_ids"], response_mask
                    )
                    token_nll = -log_probs.gather(1, targets[:, None]).squeeze(1)
                    if condition == "full":
                        full_log_probs = log_probs
                        token_kl = torch.zeros_like(token_nll)
                        if hook_parity is None:
                            controller.enabled = False
                            native = model(**inputs, use_cache=False).logits
                            controller.enabled = True
                            hook_parity = float((native.float() - logits.float()).abs().max())
                    else:
                        token_kl = (
                            full_log_probs.exp() * (full_log_probs - log_probs)
                        ).sum(dim=-1)
                rows = rows.cpu().numpy() + start
                np.add.at(nll[condition], rows, token_nll.double().cpu().numpy())
                np.add.at(kl[condition], rows, token_kl.double().cpu().numpy())
                if condition == "full":
                    np.add.at(token_counts, rows, 1.0)
            controller.collect = None
    finally:
        controller.close()

    tolerance = 1e-3 if args.dtype == "float32" else 0.5
    if hook_parity is None or hook_parity > tolerance:
        raise RuntimeError(f"Hooked 'full' forward differs from the native one: {hook_parity}.")

    keep = token_counts > 0
    rng = np.random.default_rng(args.seed)
    total = token_counts[keep].sum()
    results = {}
    for condition in conditions:
        results[condition] = {
            "nll": float(nll[condition][keep].sum() / total),
            "kl_from_full": float(kl[condition][keep].sum() / total),
            "nll_minus_zero": bootstrap_delta(
                (nll[condition] - nll["zero"])[keep], token_counts[keep], args.bootstrap, rng
            ),
            "nll_minus_full": bootstrap_delta(
                (nll[condition] - nll["full"])[keep], token_counts[keep], args.bootstrap, rng
            ),
        }

    modules = describe_modules(controller, fits, calibration, heldout)
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dtype": args.dtype,
        "fit_scope": args.fit_scope,
        "n_calibration": len(calibration_rows),
        "n_samples": int(keep.sum()),
        "n_tokens": int(total),
        "eval_offset": args.eval_offset,
        "hook_parity_max_abs_logit_diff": hook_parity,
        "conditions": results,
        "by_projection": group_means(modules, "projection"),
        "by_layer_band": group_means(modules, "layer_band"),
        "modules": modules,
        "per_sample": {
            "sample_id": [args.eval_offset + int(i) for i in np.flatnonzero(keep)],
            "n_tokens": token_counts[keep].astype(int).tolist(),
            "nll_sum": {c: nll[c][keep].tolist() for c in conditions},
            "kl_sum": {c: kl[c][keep].tolist() for c in conditions},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'condition':<18}{'NLL':>10}{'vs zero':>12}{'CI95':>26}{'vs full':>12}"
          f"{'CI95':>26}{'KL(full||.)':>14}")
    for condition, row in results.items():
        zero, full = row["nll_minus_zero"], row["nll_minus_full"]
        print(f"{condition:<18}{row['nll']:>10.5f}{zero['delta']:>+12.2e}"
              f"{'[%+.2e, %+.2e]' % tuple(zero['ci95']):>26}{full['delta']:>+12.2e}"
              f"{'[%+.2e, %+.2e]' % tuple(full['ci95']):>26}{row['kl_from_full']:>14.2e}")
    print("\nHeld-out R^2 of the correction (mean over modules):")
    for group, row in summary["by_projection"].items():
        print(f"  {group:<10} scalar={row['heldout_r2_scalar']:.3f} "
              f"diag={row['heldout_r2_diag']:.3f} matrix={row['heldout_r2_matrix']:.3f} "
              f"| gain={row['heldout_gain_ratio']:.4f} offset={row['heldout_offset_ratio']:.4f} "
              f"curvature={row['heldout_curvature_ratio_scalar']:.4f}")
    print(f"[DONE] {args.output}")


if __name__ == "__main__":
    main()
