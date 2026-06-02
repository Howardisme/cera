"""
Dataset loading and tokenization utilities for CeRA experiments.

Supported task datasets
-----------------------
  math  : TIGER-Lab/MathInstruct        (instruction-following, math reasoning)
  code  : sahil2801/CodeAlpaca-20k      (code generation)
  orca  : Open-Orca/SlimOrca            (multi-turn chat / reasoning)

Forgetting dataset
------------------
  WikiText-2 (wikitext-2-raw-v1) -- evaluated throughout training to detect
  catastrophic forgetting of general language knowledge.
"""

import gc
from typing import Dict, List, Any, Tuple

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizer


MAX_SEQ_LEN   = 512
MATH_TRAIN_N  = 100_000
MATH_TEST_OFF = 100_000
ORCA_TRAIN_N  = 100_000
ORCA_TEST_OFF = 100_000
WIKI_TRAIN_N  = 1_000
WIKI_TEST_N   = 500


# ------------------------------------------------------------------------------
# Format functions  (map a raw dataset record to a single string)
# ------------------------------------------------------------------------------

def fmt_math(sample: Dict[str, Any], eos: str) -> str:
    return (
        f"Question: {sample.get('instruction', '')}\n"
        f"Answer: {sample.get('output', '')}{eos}"
    )


def fmt_code(sample: Dict[str, Any], eos: str) -> str:
    if sample.get("input", "").strip():
        return (
            f"Question: {sample.get('instruction', '')}\n"
            f"Input: {sample.get('input', '')}\n"
            f"Answer: {sample.get('output', '')}{eos}"
        )
    return (
        f"Question: {sample.get('instruction', '')}\n"
        f"Answer: {sample.get('output', '')}{eos}"
    )


def fmt_orca(sample: Dict[str, Any], eos: str) -> str:
    convs = sample.get("conversations", [])
    human = next((c["value"] for c in convs if c["from"] == "human"), "")
    gpt   = next((c["value"] for c in convs if c["from"] == "gpt"),   "")
    return f"User: {human}\nAssistant: {gpt}{eos}"


# ------------------------------------------------------------------------------
# Tokenisation helper
# ------------------------------------------------------------------------------

def _tokenize(texts: List[str], tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    return tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )["input_ids"]


# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------

def load_task_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    max_train_samples: int = 100_000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Download, format, and tokenize a task-specific fine-tuning dataset.

    Args:
        dataset_name:     'math' | 'code' | 'orca'
        tokenizer:        HuggingFace tokenizer with pad_token set.
        max_train_samples: Cap on training set size (default 100 k).

    Returns:
        (ids_train, ids_test) -- padded token-ID tensors.
    """
    eos = tokenizer.eos_token

    if dataset_name == "math":
        print("[DATA] Loading MathInstruct (math reasoning)...")
        ds = load_dataset("TIGER-Lab/MathInstruct", split="train")
        n_needed = min(MATH_TEST_OFF + 2_000, len(ds))
        rows = list(ds.select(range(n_needed)))
        train_raw = rows[:max_train_samples]
        test_raw  = rows[MATH_TEST_OFF:]
        fmt = lambda x: fmt_math(x, eos)
        del ds

    elif dataset_name == "code":
        print("[DATA] Loading CodeAlpaca-20k (code generation)...")
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        n = len(ds)
        split = int(n * 0.9)
        train_raw = list(ds.select(range(split)))
        test_raw  = list(ds.select(range(split, n)))
        fmt = lambda x: fmt_code(x, eos)
        del ds

    elif dataset_name == "orca":
        print("[DATA] Loading SlimOrca (reasoning / chat)...")
        ds = load_dataset("Open-Orca/SlimOrca", split="train")
        n_needed = min(ORCA_TEST_OFF + 2_000, len(ds))
        rows = list(ds.select(range(n_needed)))
        train_raw = rows[:max_train_samples]
        test_raw  = rows[ORCA_TEST_OFF:]
        fmt = lambda x: fmt_orca(x, eos)
        del ds

    else:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. Choose from 'math', 'code', 'orca'."
        )

    gc.collect()

    if len(train_raw) > max_train_samples:
        train_raw = train_raw[:max_train_samples]

    print(f"[DATA] Tokenizing {dataset_name}: {len(train_raw):,} train | {len(test_raw):,} test")
    ids_train = _tokenize([fmt(x) for x in train_raw], tokenizer)
    ids_test  = _tokenize([fmt(x) for x in test_raw],  tokenizer)

    return ids_train, ids_test


def load_forgetting_dataset(
    tokenizer: PreTrainedTokenizer,
    train_size: int = WIKI_TRAIN_N,
    test_size:  int = WIKI_TEST_N,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load WikiText-2 for catastrophic-forgetting evaluation.

    Returns:
        (ids_train, ids_test)
    """
    print("[DATA] Loading WikiText-2 (forgetting analysis)...")
    ds_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ds_test  = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    train_raw = [x for x in list(ds_train)[:train_size] if len(x["text"]) > 20]
    test_raw  = [x for x in list(ds_test )[:test_size ] if len(x["text"]) > 20]

    del ds_train, ds_test
    gc.collect()

    ids_train = _tokenize([x["text"] for x in train_raw], tokenizer)
    ids_test  = _tokenize([x["text"] for x in test_raw ], tokenizer)

    return ids_train, ids_test
