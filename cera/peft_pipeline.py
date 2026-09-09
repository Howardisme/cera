"""Local CeRA interoperability helpers for the shared Trainer pipeline."""

import torch

from cera.adapters import CeRAWrapper, apply_cera


def peft_adapter_dtype(base_dtype):
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    with torch.random.fork_rng(devices=[]):
        config = LlamaConfig(
            vocab_size=16, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        )
        probe = LlamaForCausalLM(config).to(dtype=base_dtype)
        probe = get_peft_model(probe, LoraConfig(
            r=2, lora_alpha=2, target_modules=["q_proj", "v_proj"],
            bias="none", task_type="CAUSAL_LM",
        ))
        dtypes = {param.dtype for param in probe.parameters() if param.requires_grad}
    if len(dtypes) != 1:
        raise ValueError(f"PEFT probe has mixed trainable dtypes: {dtypes}")
    return dtypes.pop()


def adapter_manifest(model):
    return {
        name: {"shape": list(param.shape), "dtype": str(param.dtype).removeprefix("torch.")}
        for name, param in model.named_parameters() if param.requires_grad
    }


def validate_cera_config(config):
    if config.get("cera_format_version") != 1 or config.get("rank_mode") != "fixed":
        raise ValueError("Unsupported CeRA checkpoint format or rank mode.")
    if config.get("scale") != 1.0:
        raise ValueError("This CeRA implementation requires unit output scale.")
    if config.get("adapter_dtype") not in ("float32", "bfloat16", "float16"):
        raise ValueError("Invalid CeRA adapter dtype.")
    if config.get("model_type") != "CeRA" or config.get("rank", 0) < 1:
        raise ValueError("Invalid CeRA model type or rank.")


def load_cera_weights(model, state):
    normalized = {
        name.replace(".cera.up_proj.", ".cera.A.").replace(".cera.down_proj.", ".cera.B."): value
        for name, value in state.items()
    }
    expected = {name for name, _ in model.named_parameters() if ".cera." in name}
    missing = expected - normalized.keys()
    unexpected = normalized.keys() - expected
    if missing or unexpected or not expected:
        raise ValueError(f"CeRA adapter weights mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    return model.load_state_dict(normalized, strict=False)


def restore_cera(model, checkpoint, *, rank, dropout, act_fn, target_modules):
    config = checkpoint.get("config")
    if config is not None and "cera_format_version" in config:
        validate_cera_config(config)
        model = apply_cera(
            model, rank=config["rank"], dropout=config["dropout"],
            act_fn=config["act_fn"], target_modules=config["target_modules"].split(","),
            adapter_dtype=getattr(torch, config["adapter_dtype"]),
        )
    else:
        model = apply_cera(
            model, rank / model.config.hidden_size, dropout=dropout,
            act_fn=act_fn, target_modules=target_modules,
        )
    load_cera_weights(model, checkpoint["model_state_dict"])
    return model


def validate_cera_budget(model, rank, expected_projections):
    wrappers = [module for module in model.modules() if isinstance(module, CeRAWrapper)]
    if len(wrappers) != expected_projections:
        raise ValueError(f"Expected {expected_projections} projections, got {len(wrappers)}")
    expected = sum(rank * (module.original_layer.in_features + module.original_layer.out_features)
                   for module in wrappers)
    actual = sum(param.numel() for param in model.parameters() if param.requires_grad)
    if actual != expected or any(module.cera.A.out_features != rank for module in wrappers):
        raise ValueError(f"CeRA budget mismatch: expected={expected}, actual={actual}")
    return actual