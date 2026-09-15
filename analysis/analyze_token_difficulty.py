#!/usr/bin/env python3
"""Teacher-forced token analysis for linear versus nonlinear PEFT.

The frozen base model defines predictive difficulty. Matching LoRA and CeRA
checkpoints are then evaluated on exactly the same gold prefixes. Only response
tokens are scored; prompts are context but never observations.
"""

import argparse
import gc
import json
import os
import random
import sys
from argparse import Namespace
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import DTYPE, load_model_with_adapter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure CeRA-vs-LoRA token benefit against frozen-base difficulty."
    )
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--dataset", choices=("metamathqa", "gsm8k", "math500"),
                        default="metamathqa")
    parser.add_argument("--lora_checkpoint", required=True)
    parser.add_argument("--cera_checkpoint", required=True)
    parser.add_argument("--lora_adapter_format", choices=("peft", "legacy"),
                        default="peft")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--act_fn", choices=("silu", "relu", "identity"), default="silu")
    parser.add_argument("--target_modules", default="q_proj,v_proj")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--keep_model_files", action="store_true")
    return parser.parse_args()


def load_gold_samples(dataset_name, limit):
    if dataset_name == "metamathqa":
        dataset = load_dataset("meta-math/MetaMathQA", split="train")
        start = 40_000
        dataset = dataset.select(range(start, min(start + limit, len(dataset))))
        samples = [
            {
                "sample_id": start + index,
                "prompt": f"Question: {sample['query']}\nAnswer:",
                "completion": sample["response"],
            }
            for index, sample in enumerate(dataset)
        ]
    elif dataset_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        dataset = dataset.select(range(min(limit, len(dataset))))
        samples = [
            {
                "sample_id": index,
                "prompt": f"Question: {sample['question']}\nAnswer:",
                "completion": sample["answer"],
            }
            for index, sample in enumerate(dataset)
        ]
    else:
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        dataset = dataset.select(range(min(limit, len(dataset))))
        samples = [
            {
                "sample_id": index,
                "prompt": f"Question: {sample['problem']}\nAnswer:",
                "completion": sample["solution"],
            }
            for index, sample in enumerate(dataset)
        ]
    if not samples:
        raise ValueError("The selected dataset slice is empty.")
    return samples


def tokenizer_and_revision(args, hf_token):
    checkpoint = torch.load(args.cera_checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint.get("config") or {}
    checkpoint_model = config.get("model")
    if checkpoint_model and checkpoint_model != args.base_model:
        raise ValueError(
            f"CeRA checkpoint uses {checkpoint_model!r}, not {args.base_model!r}."
        )
    revision_kwargs = {}
    if config.get("base_model_revision"):
        revision_kwargs["revision"] = config["base_model_revision"]
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, token=hf_token, **revision_kwargs
    )
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for response offset masking.")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer, revision_kwargs


def validate_checkpoint_pair(args):
    cera_checkpoint = torch.load(
        args.cera_checkpoint, map_location="cpu", weights_only=True
    )
    cera_config = cera_checkpoint.get("config") or {}
    expected_targets = set(args.target_modules.split(","))
    for name, config in (("CeRA", cera_config),):
        if config.get("rank") is not None and config["rank"] != args.rank:
            raise ValueError(f"{name} rank mismatch: {config['rank']} != {args.rank}")
        if config.get("target_modules"):
            actual_targets = set(config["target_modules"].split(","))
            if actual_targets != expected_targets:
                raise ValueError(
                    f"{name} targets mismatch: {actual_targets} != {expected_targets}"
                )

    if args.lora_adapter_format == "peft":
        config_path = Path(args.lora_checkpoint) / "adapter_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        lora_config = json.loads(config_path.read_text(encoding="utf-8"))
        if lora_config.get("r") != args.rank:
            raise ValueError(f"LoRA rank mismatch: {lora_config.get('r')} != {args.rank}")
        if lora_config.get("lora_alpha") != args.alpha:
            raise ValueError(
                f"LoRA alpha mismatch: {lora_config.get('lora_alpha')} != {args.alpha}"
            )
        actual_targets = set(lora_config.get("target_modules") or [])
        if actual_targets != expected_targets:
            raise ValueError(
                f"LoRA targets mismatch: {actual_targets} != {expected_targets}"
            )
        checkpoint_model = lora_config.get("base_model_name_or_path")
        if checkpoint_model and checkpoint_model != args.base_model:
            raise ValueError(
                f"LoRA checkpoint uses {checkpoint_model!r}, not {args.base_model!r}."
            )
    else:
        lora_checkpoint = torch.load(
            args.lora_checkpoint, map_location="cpu", weights_only=True
        )
        lora_config = lora_checkpoint.get("config") or {}
        if lora_config.get("rank") is not None and lora_config["rank"] != args.rank:
            raise ValueError(
                f"LoRA rank mismatch: {lora_config['rank']} != {args.rank}"
            )


def encode_batch(tokenizer, samples, max_length):
    texts = []
    response_starts = []
    for sample in samples:
        prefix = sample["prompt"] + " "
        texts.append(prefix + sample["completion"] + tokenizer.eos_token)
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
    response_mask = torch.zeros_like(encoded["input_ids"], dtype=torch.bool)
    for row, response_start in enumerate(response_starts):
        response_mask[row] = (
            (offsets[row, :, 1] > response_start)
            & encoded["attention_mask"][row].bool()
        )
    return encoded, offsets, response_mask


def load_base_model(args, hf_token, revision_kwargs):
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        token=hf_token,
        torch_dtype=DTYPE,
        device_map="auto",
        attn_implementation="sdpa",
        **revision_kwargs,
    )
    model.config.use_cache = False
    model.eval()
    return model


def adapter_namespace(args, model_name):
    is_lora = model_name == "lora"
    return Namespace(
        base_model=args.base_model,
        adapter_type="lora" if is_lora else "cera",
        rank=args.rank,
        checkpoint=args.lora_checkpoint if is_lora else args.cera_checkpoint,
        adapter_format=args.lora_adapter_format if is_lora else "legacy",
        alpha=args.alpha,
        act_fn=args.act_fn,
        dropout=args.dropout,
        target_modules=args.target_modules,
        explicit_options=set(),
    )


def release_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_model(model_name, model, tokenizer, samples, args, output_path):
    device = model.get_input_embeddings().weight.device
    with output_path.open("w", encoding="utf-8") as output:
        for batch_start in tqdm(
            range(0, len(samples), args.batch_size), desc=f"Scoring {model_name}"
        ):
            batch_samples = samples[batch_start:batch_start + args.batch_size]
            encoded, offsets, response_mask = encode_batch(
                tokenizer, batch_samples, args.max_length
            )
            model_inputs = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = model(**model_inputs).logits[:, :-1].float()

            gold = encoded["input_ids"][:, 1:]
            valid = response_mask[:, 1:] & encoded["attention_mask"][:, 1:].bool()
            log_normalizer = torch.logsumexp(logits, dim=-1).cpu()
            gold_logits = logits.gather(-1, gold.to(device).unsqueeze(-1)).squeeze(-1).cpu()
            nll = log_normalizer - gold_logits
            probabilities = torch.softmax(logits, dim=-1)
            entropy = (log_normalizer.to(device) - (probabilities * logits).sum(-1)).cpu()
            top1 = logits.argmax(-1).cpu()
            del logits, probabilities

            for row, sample in enumerate(batch_samples):
                positions = torch.nonzero(valid[row], as_tuple=False).flatten().tolist()
                for prediction_position in positions:
                    token_position = prediction_position + 1
                    token_id = int(gold[row, prediction_position])
                    record = {
                        "sample_id": sample["sample_id"],
                        "token_position": token_position,
                        "token_id": token_id,
                        "token": tokenizer.decode([token_id]),
                        "char_start": int(offsets[row, token_position, 0]),
                        "char_end": int(offsets[row, token_position, 1]),
                        "nll": float(nll[row, prediction_position]),
                        "entropy": float(entropy[row, prediction_position]),
                        "top1_correct": bool(top1[row, prediction_position] == token_id),
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_record(handle, path):
    line = handle.readline()
    if not line:
        raise ValueError(f"Unexpected end of aligned model file: {path}")
    return json.loads(line)


def merge_model_files(model_paths, output_path):
    with ExitStack() as stack:
        handles = {
            name: stack.enter_context(path.open(encoding="utf-8"))
            for name, path in model_paths.items()
        }
        with output_path.open("w", encoding="utf-8") as output:
            while True:
                first_line = handles["base"].readline()
                if not first_line:
                    break
                records = {"base": json.loads(first_line)}
                records.update({
                    name: read_record(handles[name], model_paths[name])
                    for name in ("lora", "cera")
                })
                keys = {
                    (record["sample_id"], record["token_position"], record["token_id"])
                    for record in records.values()
                }
                if len(keys) != 1:
                    raise ValueError(f"Model output alignment failure: {keys}")
                base, lora, cera = records["base"], records["lora"], records["cera"]
                merged = {
                    "sample_id": base["sample_id"],
                    "token_position": base["token_position"],
                    "token_id": base["token_id"],
                    "token": base["token"],
                    "char_start": base["char_start"],
                    "char_end": base["char_end"],
                    "base_nll": base["nll"],
                    "base_entropy": base["entropy"],
                    "base_top1_correct": base["top1_correct"],
                    "lora_nll": lora["nll"],
                    "cera_nll": cera["nll"],
                    "lora_top1_correct": lora["top1_correct"],
                    "cera_top1_correct": cera["top1_correct"],
                    "cera_benefit": lora["nll"] - cera["nll"],
                }
                output.write(json.dumps(merged, ensure_ascii=False) + "\n")
            for name in ("lora", "cera"):
                if handles[name].readline():
                    raise ValueError(f"Extra records in {model_paths[name]}")


def rank_values(values):
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def summarize(combined_path, bootstrap, seed):
    base_nll = []
    benefit = []
    sample_ids = []
    records = []
    with combined_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records.append(record)
            base_nll.append(record["base_nll"])
            benefit.append(record["cera_benefit"])
            sample_ids.append(record["sample_id"])
    if not records:
        raise ValueError("No completion tokens were scored.")

    base_nll_array = np.asarray(base_nll, dtype=np.float64)
    benefit_array = np.asarray(benefit, dtype=np.float64)
    boundaries = np.quantile(base_nll_array, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(boundaries[1:-1], base_nll_array, side="right"), 0, 9)
    unique_samples = sorted(set(sample_ids))
    sample_index = {sample_id: index for index, sample_id in enumerate(unique_samples)}
    cluster_sums = np.zeros((len(unique_samples), 10), dtype=np.float64)
    cluster_counts = np.zeros((len(unique_samples), 10), dtype=np.int64)
    for record, bin_index in zip(records, bins):
        row = sample_index[record["sample_id"]]
        cluster_sums[row, bin_index] += record["cera_benefit"]
        cluster_counts[row, bin_index] += 1

    rng = np.random.default_rng(seed)
    bootstrap_means = np.full((bootstrap, 10), np.nan, dtype=np.float64)
    for draw in range(bootstrap):
        selected = rng.integers(0, len(unique_samples), size=len(unique_samples))
        sums = cluster_sums[selected].sum(axis=0)
        counts = cluster_counts[selected].sum(axis=0)
        bootstrap_means[draw] = np.divide(
            sums, counts, out=np.full(10, np.nan), where=counts > 0
        )

    bin_rows = []
    for bin_index in range(10):
        selected_records = [
            record for record, assigned_bin in zip(records, bins)
            if assigned_bin == bin_index
        ]
        values = np.asarray(
            [record["cera_benefit"] for record in selected_records], dtype=np.float64
        )
        draws = bootstrap_means[:, bin_index]
        bin_rows.append({
            "difficulty_decile": bin_index + 1,
            "n_tokens": len(selected_records),
            "base_nll_mean": float(np.mean([
                record["base_nll"] for record in selected_records
            ])),
            "cera_benefit_mean": float(values.mean()),
            "cera_benefit_median": float(np.median(values)),
            "cera_token_win_rate": float(np.mean(values > 0)),
            "bootstrap_ci95": [
                float(np.nanquantile(draws, 0.025)),
                float(np.nanquantile(draws, 0.975)),
            ],
            "base_top1_accuracy": float(np.mean([
                record["base_top1_correct"] for record in selected_records
            ])),
            "lora_top1_accuracy": float(np.mean([
                record["lora_top1_correct"] for record in selected_records
            ])),
            "cera_top1_accuracy": float(np.mean([
                record["cera_top1_correct"] for record in selected_records
            ])),
            "lora_wrong_cera_correct": int(sum(
                not record["lora_top1_correct"] and record["cera_top1_correct"]
                for record in selected_records
            )),
            "lora_correct_cera_wrong": int(sum(
                record["lora_top1_correct"] and not record["cera_top1_correct"]
                for record in selected_records
            )),
        })

    correlation = float(np.corrcoef(
        rank_values(base_nll_array), rank_values(benefit_array)
    )[0, 1])
    return {
        "n_samples": len(unique_samples),
        "n_tokens": len(records),
        "spearman_base_nll_vs_cera_benefit": correlation,
        "overall_cera_benefit_mean": float(benefit_array.mean()),
        "difficulty_boundaries": boundaries.tolist(),
        "bins": bin_rows,
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.num_samples < 1 or args.batch_size < 1 or args.bootstrap < 1:
        raise ValueError("num_samples, batch_size, and bootstrap must be positive.")
    if not Path(args.cera_checkpoint).is_file():
        raise FileNotFoundError(args.cera_checkpoint)
    lora_path = Path(args.lora_checkpoint)
    if args.lora_adapter_format == "peft" and not lora_path.is_dir():
        raise FileNotFoundError(f"Expected PEFT adapter directory: {lora_path}")
    if args.lora_adapter_format == "legacy" and not lora_path.is_file():
        raise FileNotFoundError(lora_path)
    validate_checkpoint_pair(args)

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN is not set.")

    samples = load_gold_samples(args.dataset, args.num_samples)
    tokenizer, revision_kwargs = tokenizer_and_revision(args, hf_token)
    model_paths = {
        name: args.output_dir / f"{name}_tokens.jsonl"
        for name in ("base", "lora", "cera")
    }

    base_model = load_base_model(args, hf_token, revision_kwargs)
    score_model("base", base_model, tokenizer, samples, args, model_paths["base"])
    release_model(base_model)

    for model_name in ("lora", "cera"):
        model, model_tokenizer = load_model_with_adapter(
            adapter_namespace(args, model_name), hf_token
        )
        model_tokenizer.padding_side = "right"
        if model_tokenizer.get_vocab() != tokenizer.get_vocab():
            raise ValueError(f"Tokenizer mismatch for {model_name} checkpoint.")
        score_model(
            model_name, model, model_tokenizer, samples, args, model_paths[model_name]
        )
        release_model(model)

    combined_path = args.output_dir / "tokens.jsonl"
    merge_model_files(model_paths, combined_path)
    summary = summarize(combined_path, args.bootstrap, args.seed)
    summary["config"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[DONE] Token records: {combined_path}")
    print(f"[DONE] Summary: {summary_path}")

    if not args.keep_model_files:
        for path in model_paths.values():
            path.unlink()


if __name__ == "__main__":
    main()