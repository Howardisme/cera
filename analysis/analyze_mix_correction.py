#!/usr/bin/env python3
"""Measure learned-mix correction magnitude on held-out MetaMathQA tokens."""

import argparse
import json
import os
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cera.adapters import CeRAWrapper
from evaluate import load_model_with_adapter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure ||gamma(SiLU(z)-z)|| / ||z|| for learned-mix CeRA."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--scope", choices=("response", "all"), default="response")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def encode_batch(tokenizer, samples, max_length, scope):
    texts = []
    response_starts = []
    for sample in samples:
        prefix = f"Question: {sample['query']}\nAnswer: "
        texts.append(prefix + sample["response"] + tokenizer.eos_token)
        response_starts.append(len(prefix))
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        return_offsets_mapping=True,
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    offsets = encoded.pop("offset_mapping")
    mask = encoded["attention_mask"].bool()
    if scope == "response":
        for row, response_start in enumerate(response_starts):
            mask[row] &= offsets[row, :, 1] > response_start
    return encoded, mask


class CorrectionTracker:
    def __init__(self):
        self.mask = None
        self.stats = {}
        self.handles = []

    def attach(self, model):
        for name, module in model.named_modules():
            if not isinstance(module, CeRAWrapper):
                continue
            if module.cera.mix_mode != "learned_mix":
                raise ValueError(f"{name} is not a learned-mix CeRA adapter.")
            self.stats[name] = {
                "z_sq": None,
                "correction_sq": None,
                "mixed_sq": None,
                "negative": 0,
                "elements": 0,
                "gamma": float(module.cera.gamma.detach().cpu()),
            }
            self.handles.append(
                module.cera.A.register_forward_hook(self._hook(name, module.cera))
            )
        if not self.stats:
            raise ValueError("No CeRA adapters found in model.")

    def _hook(self, name, adapter):
        def record(_module, _inputs, output):
            mask = self.mask.to(output.device)
            selected = output.detach()[mask].float()
            if selected.numel() == 0:
                return
            correction = adapter.gamma.detach().float() * (F.silu(selected) - selected)
            mixed = selected + correction
            current = self.stats[name]
            values = {
                "z_sq": selected.square().sum(),
                "correction_sq": correction.square().sum(),
                "mixed_sq": mixed.square().sum(),
            }
            for key, value in values.items():
                current[key] = value if current[key] is None else current[key] + value
            current["negative"] += int((selected < 0).sum())
            current["elements"] += selected.numel()
        return record

    def close(self):
        for handle in self.handles:
            handle.remove()


def summarize(tracker, checkpoint, scope, n_samples):
    modules = []
    for name, stats in tracker.stats.items():
        if not stats["elements"]:
            continue
        z_sq = float(stats["z_sq"].cpu())
        correction_sq = float(stats["correction_sq"].cpu())
        mixed_sq = float(stats["mixed_sq"].cpu())
        layer = int(name.split(".layers.", 1)[1].split(".", 1)[0])
        projection = next(
            projection for projection in ("q_proj", "v_proj", "k_proj", "o_proj",
                                           "gate_proj", "up_proj", "down_proj")
            if f".{projection}" in name
        )
        modules.append({
            "module": name,
            "layer": layer,
            "projection": projection,
            "gamma": stats["gamma"],
            "elements": stats["elements"],
            "negative_fraction": stats["negative"] / stats["elements"],
            "correction_ratio": (correction_sq / z_sq) ** 0.5,
            "mixed_to_linear_ratio": (mixed_sq / z_sq) ** 0.5,
            "z_rms": (z_sq / stats["elements"]) ** 0.5,
        })

    def grouped(rows, key):
        groups = defaultdict(list)
        for row in rows:
            groups[row[key]].append(row)
        return {
            str(group): {
                "count": len(group_rows),
                "mean_gamma": sum(row["gamma"] for row in group_rows) / len(group_rows),
                "mean_correction_ratio": sum(
                    row["correction_ratio"] for row in group_rows
                ) / len(group_rows),
                "mean_mixed_to_linear_ratio": sum(
                    row["mixed_to_linear_ratio"] for row in group_rows
                ) / len(group_rows),
            }
            for group, group_rows in groups.items()
        }

    for row in modules:
        row["layer_band"] = (
            "early" if row["layer"] <= 7
            else "middle" if row["layer"] <= 23
            else "late"
        )
    return {
        "checkpoint": str(checkpoint.resolve()),
        "scope": scope,
        "n_samples": n_samples,
        "n_modules": len(modules),
        "by_projection": grouped(modules, "projection"),
        "by_layer_band": grouped(modules, "layer_band"),
        "modules": modules,
    }


def main():
    args = parse_args()
    if args.num_samples < 1 or args.batch_size < 1 or args.max_length < 2:
        raise ValueError("num_samples, batch_size, and max_length must be positive.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint.get("config") or {}
    if config.get("cera_mix_mode") != "learned_mix":
        raise ValueError("Checkpoint is not learned-mix CeRA.")

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN is not set.")
    namespace = Namespace(
        base_model=config["model"], adapter_type="cera", rank=config["rank"],
        checkpoint=str(args.checkpoint), adapter_format="legacy",
        alpha=config["alpha"], act_fn=config["act_fn"], dropout=config["dropout"],
        target_modules=config["target_modules"], explicit_options=set(),
    )
    model, tokenizer = load_model_with_adapter(namespace, hf_token)
    tokenizer.padding_side = "right"
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for response masking.")

    dataset = load_dataset("meta-math/MetaMathQA", split="train")
    start = 40_000
    samples = list(dataset.select(range(start, min(start + args.num_samples, len(dataset)))))
    tracker = CorrectionTracker()
    tracker.attach(model)
    device = model.get_input_embeddings().weight.device
    try:
        for batch_start in tqdm(range(0, len(samples), args.batch_size), desc="Probing"):
            encoded, mask = encode_batch(
                tokenizer,
                samples[batch_start:batch_start + args.batch_size],
                args.max_length,
                args.scope,
            )
            tracker.mask = mask
            model_inputs = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                model(**model_inputs, use_cache=False)
    finally:
        tracker.close()

    summary = summarize(tracker, args.checkpoint, args.scope, len(samples))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "checkpoint": summary["checkpoint"],
        "scope": summary["scope"],
        "n_samples": summary["n_samples"],
        "n_modules": summary["n_modules"],
        "by_projection": summary["by_projection"],
        "by_layer_band": summary["by_layer_band"],
    }, indent=2))
    print(f"[DONE] {args.output}")


if __name__ == "__main__":
    main()