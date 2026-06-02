#!/usr/bin/env python3
"""
compare_generations.py -- Qualitative generation comparison: CeRA vs LoRA.

Loads two fine-tuned checkpoints, generates answers for a MathInstruct
test slice, and writes a side-by-side comparison to an output text file.

Example:
    python analysis/compare_generations.py \\
        --cera_ckpt results/Exp_CeRA_math_.../CeRA/cera_ckpt_best_75000.pt \\
        --lora_ckpt results/Exp_LoRA_math_.../LoRA/lora_ckpt_best_75000.pt \\
        --cera_rank 128 --lora_rank 128
"""

import argparse
import os
import sys

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cera.adapters import CeRAWrapper, LoRAWrapper


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE         = torch.bfloat16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate CeRA vs LoRA qualitative comparisons on MathInstruct.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base_model", default=DEFAULT_MODEL,
                   help="HuggingFace model identifier.")
    p.add_argument("--cera_ckpt", required=True,
                   help="Path to the CeRA checkpoint (.pt file).")
    p.add_argument("--lora_ckpt", required=True,
                   help="Path to the LoRA checkpoint (.pt file).")
    p.add_argument("--cera_rank",   type=int, default=128,
                   help="CeRA rank (used to derive expansion_factor).")
    p.add_argument("--lora_rank",   type=int, default=128,
                   help="LoRA rank.")
    p.add_argument("--num_samples", type=int, default=100,
                   help="Number of test examples to generate.")
    p.add_argument("--output", default="results/case_study_candidates.txt",
                   help="Output file path.")
    return p.parse_args()


def load_model_with_adapter(ckpt_path: str, model_type: str, rank: int,
                             base_model: str, hf_token: str) -> torch.nn.Module:
    print(f"[INFO] Loading {model_type} from {ckpt_path}...")

    model = AutoModelForCausalLM.from_pretrained(
        base_model, token=hf_token, torch_dtype=DTYPE, device_map=DEVICE
    )

    ckpt       = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    state_dict = ckpt["model_state_dict"]

    target_modules = ["q_proj", "v_proj"]
    print(f"[INFO] Injecting {model_type} adapters...")

    for layer in model.model.layers:
        for name, module in layer.self_attn.named_children():
            if name not in target_modules:
                continue
            if model_type == "CeRA":
                atten_dim = module.in_features
                exp       = rank / atten_dim
                wrapper   = CeRAWrapper(module, module.in_features, module.out_features, exp)
            else:
                wrapper = LoRAWrapper(module, module.in_features, module.out_features, rank)
            wrapper.to(device=DEVICE, dtype=DTYPE)
            setattr(layer.self_attn, name, wrapper)

    keys = model.load_state_dict(state_dict, strict=False)
    print(
        f"[INFO] Weights loaded. "
        f"(Unexpected: {len(keys.unexpected_keys)}, Missing: {len(keys.missing_keys)})"
    )
    model.eval()
    return model


def generate_answers(model, tokenizer, prompts):
    outputs = []
    with torch.no_grad():
        for p in tqdm(prompts):
            inputs = tokenizer(p, return_tensors="pt").to(DEVICE)
            out    = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            outputs.append(tokenizer.decode(out[0], skip_special_tokens=True))
    return outputs


def main():
    args = parse_args()

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN", "")

    for path, label in [(args.cera_ckpt, "CeRA"), (args.lora_ckpt, "LoRA")]:
        if not os.path.exists(path):
            print(f"[WARNING] {label} checkpoint not found: {path}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Loading MathInstruct test slice...")
    ds        = load_dataset("TIGER-Lab/MathInstruct", split="train")
    test_data = ds.select(range(100_000, 100_000 + args.num_samples))
    prompts   = [f"Question: {x['instruction']}\nAnswer:" for x in test_data]
    golds     = [x["output"] for x in test_data]

    # CeRA generation
    if os.path.exists(args.cera_ckpt):
        model_cera   = load_model_with_adapter(
            args.cera_ckpt, "CeRA", args.cera_rank, args.base_model, hf_token
        )
        cera_outputs = generate_answers(model_cera, tokenizer, prompts)
        del model_cera
        torch.cuda.empty_cache()
    else:
        cera_outputs = ["(checkpoint not found)"] * len(prompts)

    # LoRA generation
    if os.path.exists(args.lora_ckpt):
        model_lora   = load_model_with_adapter(
            args.lora_ckpt, "LoRA", args.lora_rank, args.base_model, hf_token
        )
        lora_outputs = generate_answers(model_lora, tokenizer, prompts)
        del model_lora
        torch.cuda.empty_cache()
    else:
        lora_outputs = ["(checkpoint not found)"] * len(prompts)

    # Write comparison file
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for i in range(len(prompts)):
            cera_ans = cera_outputs[i].replace(prompts[i], "").strip()
            lora_ans = lora_outputs[i].replace(prompts[i], "").strip()

            f.write("=" * 80 + "\n")
            f.write(f"Case ID: {i}\n")
            f.write(f"[Question]:\n{test_data[i]['instruction']}\n\n")
            f.write(f"[Gold Answer]:\n{golds[i]}\n\n")
            f.write("-" * 40 + "\n")
            f.write(f"[CeRA (R{args.cera_rank})]:\n{cera_ans}\n\n")
            f.write("-" * 40 + "\n")
            f.write(f"[LoRA (R{args.lora_rank})]:\n{lora_ans}\n\n")
            f.write("=" * 80 + "\n\n")

    print(f"[SUCCESS] Saved -> {args.output}")


if __name__ == "__main__":
    main()
