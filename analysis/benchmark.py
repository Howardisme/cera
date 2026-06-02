#!/usr/bin/env python3
"""
benchmark.py -- Inference throughput / latency benchmark.

Measures token generation speed for CeRA, LoRA, and DoRA adapter
configurations on a fixed base model and writes results to a CSV file.

Example:
    python analysis/benchmark.py \\
        --base_model meta-llama/Llama-3.1-8B \\
        --output_csv results/efficiency_results.csv
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cera.adapters import CeRAWrapper, LoRAWrapper


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE         = torch.bfloat16
NUM_WARMUP    = 5
NUM_TESTS     = 20
GEN_TOKENS    = 50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference throughput benchmark for CeRA / LoRA adapters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base_model", default=DEFAULT_MODEL,
                   help="HuggingFace model identifier.")
    p.add_argument("--output_csv", default="results/efficiency_results.csv",
                   help="Output CSV file path.")
    return p.parse_args()


def inject_adapter(model, model_type: str, rank: int):
    """Replace (or swap) the adapter on an already-loaded model."""
    target_modules = ["q_proj", "v_proj"]
    trainable = 0

    for layer in model.model.layers:
        for name, module in layer.self_attn.named_children():
            if name not in target_modules:
                continue

            original = module.original_layer if isinstance(module, (CeRAWrapper, LoRAWrapper)) else module
            atten_dim = original.in_features

            if model_type == "CeRA":
                exp     = rank / atten_dim
                wrapper = CeRAWrapper(original, original.in_features, original.out_features, exp)
                trainable += sum(p.numel() for p in wrapper.cera.parameters())
            else:
                wrapper = LoRAWrapper(original, original.in_features, original.out_features, rank)
                trainable += wrapper.lora_A.numel() + wrapper.lora_B.numel()

            wrapper.to(DEVICE, dtype=DTYPE)
            setattr(layer.self_attn, name, wrapper)

    return model, trainable


def benchmark(model, tokenizer):
    model.eval()
    dummy = tokenizer("Hello, mathematical reasoning is", return_tensors="pt").to(DEVICE)

    print("   Warming up GPU...")
    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            model.generate(**dummy, max_new_tokens=GEN_TOKENS,
                           min_new_tokens=GEN_TOKENS, do_sample=False)

    print(f"   Running {NUM_TESTS} timed trials...")
    latencies = []
    with torch.no_grad():
        for _ in range(NUM_TESTS):
            torch.cuda.synchronize()
            t0 = time.time()
            model.generate(**dummy, max_new_tokens=GEN_TOKENS,
                           min_new_tokens=GEN_TOKENS, do_sample=False)
            torch.cuda.synchronize()
            latencies.append(time.time() - t0)

    avg_ms_per_tok = (np.mean(latencies) / GEN_TOKENS) * 1000
    tps            = GEN_TOKENS / np.mean(latencies)
    return avg_ms_per_tok, tps


def main():
    args = parse_args()

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    print(f"[INFO] Benchmarking on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
    model     = AutoModelForCausalLM.from_pretrained(
        args.base_model, token=hf_token, torch_dtype=DTYPE, device_map=DEVICE
    )
    model.config.use_cache = True

    configs = [
        {"model_type": "LoRA", "rank": 512},
        {"model_type": "CeRA", "rank": 64},
        {"model_type": "CeRA", "rank": 128},
    ]
    results      = []
    base_latency = None

    for cfg in configs:
        m_type = cfg["model_type"]
        rank   = cfg["rank"]
        print(f"\n[BENCH] {m_type} R={rank}")

        model, params = inject_adapter(model, m_type, rank)
        lat_ms, tps   = benchmark(model, tokenizer)

        if base_latency is None:
            base_latency = lat_ms
            rel          = 1.00
        else:
            rel = lat_ms / base_latency

        results.append({
            "Model":              m_type,
            "Rank":               rank,
            "Params (M)":         f"{params / 1e6:.2f}M",
            "Latency (ms/tok)":   f"{lat_ms:.2f}",
            "Throughput (tok/s)": f"{tps:.2f}",
            "Rel. Latency":       f"{rel:.2f}x",
        })

    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("EFFICIENCY RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))

    out_path = args.output_csv
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[INFO] Saved -> {out_path}")


if __name__ == "__main__":
    main()
