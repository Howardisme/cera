import copy
import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch
from torch import nn

from cera.adapters import CeRAWrapper, apply_cera
from cera.peft_pipeline import (
    load_cera_weights,
    peft_adapter_dtype,
    restore_cera,
    validate_cera_budget,
    validate_cera_config,
)
from cera.trainer import save_checkpoint


TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def tiny_model(dtype=torch.float32):
    model = nn.Module()
    model.model = nn.Module()
    layer = nn.Module()
    layer.self_attn = nn.ModuleDict({
        name: nn.Linear(8, 4 if name in ("k_proj", "v_proj") else 8, bias=False, dtype=dtype)
        for name in TARGETS[:4]
    })
    layer.mlp = nn.ModuleDict({
        "gate_proj": nn.Linear(8, 28, bias=False, dtype=dtype),
        "up_proj": nn.Linear(8, 28, bias=False, dtype=dtype),
        "down_proj": nn.Linear(28, 8, bias=False, dtype=dtype),
    })
    model.model.layers = nn.ModuleList([layer])
    model.requires_grad_(False)
    return model


class InjectionTests(unittest.TestCase):
    def test_qv_only_peft_and_cera_scope_budget_and_reload(self):
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import LlamaConfig, LlamaForCausalLM

        targets = ["q_proj", "v_proj"]
        base = LlamaForCausalLM(LlamaConfig(
            vocab_size=32, hidden_size=8, intermediate_size=28, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1,
        ))
        base.requires_grad_(False)
        pristine = copy.deepcopy(base.state_dict())
        cera = apply_cera(copy.deepcopy(base), rank=2, target_modules=targets, dropout=0)
        lora = get_peft_model(copy.deepcopy(base), LoraConfig(
            r=2, lora_alpha=2, lora_dropout=0, target_modules=targets,
            bias="none", task_type="CAUSAL_LM",
        ))
        inputs = torch.tensor([[1, 2, 3, 4]])
        directory = Path(tempfile.mkdtemp(prefix="cera-qv-test-"))
        config = dict(cera_format_version=1, rank_mode="fixed", model_type="CeRA",
                      scale=1.0, rank=2, dropout=0.0, act_fn="silu",
                      adapter_dtype="float32", target_modules=",".join(targets))
        for method, model in (("CeRA", cera), ("LoRA", lora)):
            with self.subTest(method=method):
                adapted = [(name, module) for name, module in model.named_modules()
                           if isinstance(module, CeRAWrapper) or hasattr(module, "lora_A")]
                self.assertEqual(len(adapted), 4)
                self.assertEqual({name.rsplit(".", 1)[1] for name, _ in adapted}, set(targets))
                self.assertEqual(sum(param.numel() for param in model.parameters() if param.requires_grad),
                                 2 * 2 * ((8 + 8) + (8 + 4)))
                optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=0.01)
                model.train()
                for step in range(2):
                    optimizer.zero_grad()
                    loss = model(inputs, labels=inputs).loss
                    self.assertTrue(torch.isfinite(loss).item())
                    loss.backward()
                    optimizer.step()
                inner = model if method == "CeRA" else model.get_base_model()
                for layer_index, layer in enumerate(inner.model.layers):
                    for group_name, group in (("self_attn", layer.self_attn), ("mlp", layer.mlp)):
                        for name, module in group.named_children():
                            if isinstance(module, nn.Linear):
                                key = f"model.layers.{layer_index}.{group_name}.{name}.weight"
                                torch.testing.assert_close(module.weight, pristine[key], rtol=0, atol=0)
                if method == "CeRA":
                    save_checkpoint(model, 2, 2, {}, str(directory), method, config=config)
                    checkpoint = torch.load(directory / "cera_ckpt_2.pt", weights_only=True)
                    restored = restore_cera(copy.deepcopy(base), checkpoint, rank=2,
                                            dropout=0, act_fn="silu", target_modules=targets)
                else:
                    model.save_pretrained(directory / "lora")
                    restored = PeftModel.from_pretrained(copy.deepcopy(base), directory / "lora")
                model.eval()
                restored.eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(inputs).logits, restored(inputs).logits)
        print(f"[TEST ARTIFACT] {directory}")

    def test_all_targets_fixed_rank_and_budget(self):
        model = tiny_model()
        expected = sum(2 * (module.in_features + module.out_features)
                       for module in model.modules() if isinstance(module, nn.Linear))
        apply_cera(model, rank=2, target_modules=TARGETS)
        wrappers = [module for module in model.modules() if isinstance(module, CeRAWrapper)]
        self.assertEqual(len(wrappers), 7)
        self.assertTrue(all(module.cera.A.out_features == 2 for module in wrappers))
        self.assertEqual(sum(param.numel() for param in model.parameters() if param.requires_grad), expected)
        self.assertTrue(all(".cera." in name for name, param in model.named_parameters() if param.requires_grad))

    def test_legacy_attention_only(self):
        model = apply_cera(tiny_model(), 0.25, target_modules=TARGETS)
        self.assertEqual(sum(isinstance(module, CeRAWrapper) for module in model.modules()), 4)
        self.assertIsInstance(model.model.layers[0].mlp["down_proj"], nn.Linear)

    def test_invalid_target_is_atomic_and_duplicate_injection_fails(self):
        model = tiny_model()
        with self.assertRaises(ValueError):
            apply_cera(model, rank=2, target_modules=TARGETS + ["missing"])
        self.assertFalse(any(isinstance(module, CeRAWrapper) for module in model.modules()))
        apply_cera(model, rank=2, target_modules=TARGETS)
        with self.assertRaises(ValueError):
            apply_cera(model, rank=2, target_modules=TARGETS)

    def test_zero_init_gradients_and_eval_dropout(self):
        base = nn.Linear(8, 8, bias=False)
        wrapper = CeRAWrapper(base, 8, 8, 0.25, dropout=0.1, rank=2)
        inputs = torch.randn(4, 8)
        torch.testing.assert_close(wrapper(inputs), base(inputs))
        optimizer = torch.optim.AdamW(wrapper.cera.parameters(), lr=0.01)
        wrapper(inputs).square().mean().backward()
        self.assertEqual(wrapper.cera.A.weight.grad.abs().sum().item(), 0)
        self.assertGreater(wrapper.cera.B.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad()
        wrapper(inputs).square().mean().backward()
        self.assertGreater(wrapper.cera.A.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(base.weight.grad)
        wrapper.eval()
        torch.testing.assert_close(wrapper(inputs), wrapper(inputs), rtol=0, atol=0)

    def test_learned_mix_endpoints_rng_and_gradients(self):
        base = nn.Linear(8, 8, bias=False)
        torch.manual_seed(17)
        identity = CeRAWrapper(
            copy.deepcopy(base), 8, 8, 0.25, dropout=0, act_fn="identity",
            rank=2, variant="peft_aligned",
        )
        rng_after_identity = torch.get_rng_state().clone()
        torch.manual_seed(17)
        mixed = CeRAWrapper(
            copy.deepcopy(base), 8, 8, 0.25, dropout=0, act_fn="silu",
            rank=2, variant="peft_aligned", mix_mode="learned_mix", gamma_init=0,
        )
        self.assertTrue(torch.equal(rng_after_identity, torch.get_rng_state()))
        torch.manual_seed(17)
        silu = CeRAWrapper(
            copy.deepcopy(base), 8, 8, 0.25, dropout=0, act_fn="silu",
            rank=2, variant="peft_aligned",
        )
        with torch.no_grad():
            nn.init.normal_(identity.cera.B.weight)
            for wrapper in (mixed, silu):
                wrapper.cera.A.weight.copy_(identity.cera.A.weight)
                wrapper.cera.B.weight.copy_(identity.cera.B.weight)

        inputs = torch.randn(4, 8)
        mixed.eval()
        identity.eval()
        silu.eval()
        torch.testing.assert_close(mixed(inputs), identity(inputs))
        with torch.no_grad():
            mixed.cera.gamma.fill_(1)
        torch.testing.assert_close(mixed(inputs), silu(inputs))

        gradient_wrapper = CeRAWrapper(
            copy.deepcopy(base), 8, 8, 0.25, dropout=0, act_fn="silu",
            rank=2, variant="peft_aligned", mix_mode="learned_mix", gamma_init=0,
        )
        optimizer = torch.optim.SGD(gradient_wrapper.cera.parameters(), lr=0.1)
        gradient_wrapper(inputs).square().mean().backward()
        self.assertEqual(gradient_wrapper.cera.gamma.grad.abs().item(), 0)
        self.assertEqual(gradient_wrapper.cera.A.weight.grad.abs().sum().item(), 0)
        self.assertGreater(gradient_wrapper.cera.B.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad()
        gradient_wrapper(inputs).square().mean().backward()
        self.assertGreater(gradient_wrapper.cera.gamma.grad.abs().item(), 0)
        self.assertGreater(gradient_wrapper.cera.A.weight.grad.abs().sum().item(), 0)

        with self.assertRaisesRegex(ValueError, "Learned-mix CeRA requires"):
            CeRAWrapper(
                copy.deepcopy(base), 8, 8, 0.25, act_fn="silu", rank=2,
                variant="legacy", mix_mode="learned_mix",
            )

    def test_gamma_interventions_preserve_expected_invariants(self):
        import evaluate

        base = tiny_model()
        model = apply_cera(
            copy.deepcopy(base), rank=2, dropout=0, act_fn="silu",
            target_modules=TARGETS, variant="peft_aligned", alpha=2,
            mix_mode="learned_mix", gamma_init=0,
        )
        named = sorted(
            (
                (name, module.cera.gamma)
                for name, module in model.named_modules()
                if isinstance(module, CeRAWrapper)
            ),
            key=lambda item: item[0],
        )
        with torch.no_grad():
            for index, (_, gamma) in enumerate(named):
                gamma.fill_(index + 1)
        original = [gamma.item() for _, gamma in named]

        zero_model = copy.deepcopy(model)
        zero_summary = evaluate.apply_gamma_intervention(zero_model, "zero")
        self.assertEqual(zero_summary["count"], len(original))
        self.assertTrue(all(
            module.cera.gamma.item() == 0
            for module in zero_model.modules() if isinstance(module, CeRAWrapper)
        ))

        mean_model = copy.deepcopy(model)
        evaluate.apply_gamma_intervention(mean_model, "mean")
        expected_mean = sum(original) / len(original)
        self.assertTrue(all(
            module.cera.gamma.item() == expected_mean
            for module in mean_model.modules() if isinstance(module, CeRAWrapper)
        ))

        shuffled_models = [copy.deepcopy(model), copy.deepcopy(model)]
        shuffled_values = []
        for shuffled_model in shuffled_models:
            evaluate.apply_gamma_intervention(shuffled_model, "shuffle", shuffle_seed=17)
            shuffled_values.append(sorted(
                (
                    (name, module.cera.gamma.item())
                    for name, module in shuffled_model.named_modules()
                    if isinstance(module, CeRAWrapper)
                ),
                key=lambda item: item[0],
            ))
        self.assertEqual(shuffled_values[0], shuffled_values[1])
        self.assertEqual(sorted(value for _, value in shuffled_values[0]), sorted(original))
        self.assertNotEqual([value for _, value in shuffled_values[0]], original)

    def test_fp32_adapter_bf16_base_without_autocast(self):
        base = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
        wrapper = CeRAWrapper(base, 8, 8, 0.25, rank=2, adapter_dtype=torch.float32)
        output = wrapper(torch.randn(4, 8, dtype=torch.bfloat16))
        self.assertEqual(output.dtype, torch.bfloat16)
        output.float().square().mean().backward()
        self.assertEqual(wrapper.cera.B.weight.grad.dtype, torch.float32)

    def test_legacy_renamed_keys(self):
        model = apply_cera(tiny_model(), 0.25)
        state = {name.replace(".cera.A.", ".cera.up_proj.").replace(".cera.B.", ".cera.down_proj."): param
                 for name, param in model.state_dict().items()}
        copy.deepcopy(model).load_state_dict(state, strict=True)

    def test_checkpoint_round_trip_and_missing_weight(self):
        base = tiny_model()
        model = apply_cera(copy.deepcopy(base), rank=2, target_modules=TARGETS)
        for module in model.modules():
            if isinstance(module, CeRAWrapper):
                nn.init.normal_(module.cera.B.weight)
        config = dict(cera_format_version=1, rank_mode="fixed", model_type="CeRA",
                      scale=1.0, rank=2, dropout=0.1, act_fn="silu",
                      adapter_dtype="float32", target_modules=",".join(TARGETS))
        directory = tempfile.mkdtemp(prefix="cera-checkpoint-test-")
        save_checkpoint(model, 1, 64, {}, directory, "CeRA", config=config)
        checkpoint = torch.load(Path(directory) / "cera_ckpt_64.pt", weights_only=True)
        restored = restore_cera(base, checkpoint, rank=2, dropout=0.1,
                                act_fn="silu", target_modules=TARGETS)
        model.eval()
        restored.eval()
        inputs = torch.randn(3, 28)
        torch.testing.assert_close(model.model.layers[0].mlp["down_proj"](inputs),
                                   restored.model.layers[0].mlp["down_proj"](inputs))
        checkpoint["model_state_dict"].pop(next(iter(checkpoint["model_state_dict"])))
        with self.assertRaises(ValueError):
            load_cera_weights(restored, checkpoint["model_state_dict"])
        print(f"[TEST ARTIFACT] {directory}")

    def test_peft_identity_parity_and_dtype_probe(self):
        from peft import LoraConfig, get_peft_model

        rng_state = torch.get_rng_state().clone()
        dtype = peft_adapter_dtype(torch.bfloat16)
        self.assertTrue(torch.equal(rng_state, torch.get_rng_state()))
        self.assertIn(dtype, (torch.float32, torch.bfloat16))
        base = tiny_model()
        cera = apply_cera(copy.deepcopy(base), rank=2, dropout=0,
                          act_fn="identity", target_modules=TARGETS)
        lora = get_peft_model(base, LoraConfig(r=2, lora_alpha=2, target_modules=TARGETS))
        cera_layer = cera.model.layers[0].mlp["down_proj"]
        lora_layer = lora.base_model.model.model.layers[0].mlp["down_proj"]
        with torch.no_grad():
            nn.init.normal_(cera_layer.cera.B.weight)
            lora_layer.lora_A["default"].weight.copy_(cera_layer.cera.A.weight)
            lora_layer.lora_B["default"].weight.copy_(cera_layer.cera.B.weight)
        inputs = torch.randn(3, 28)
        torch.testing.assert_close(cera_layer(inputs), lora_layer(inputs), atol=1e-5, rtol=1e-5)
        cera_layer(inputs).sum().backward()
        lora_layer(inputs).sum().backward()
        torch.testing.assert_close(cera_layer.cera.A.weight.grad, lora_layer.lora_A["default"].weight.grad)
        self.assertEqual(sum(param.numel() for param in cera.parameters() if param.requires_grad),
                         sum(param.numel() for param in lora.parameters() if param.requires_grad))

    def test_recurrent_zero_step_parity_and_initial_identity(self):
        base = tiny_model()
        torch.manual_seed(17)
        baseline = apply_cera(copy.deepcopy(base), rank=2, dropout=0,
                              act_fn="identity", target_modules=TARGETS,
                              variant="peft_aligned", alpha=2, recurrent_steps=0)
        rng_after_baseline = torch.get_rng_state().clone()
        torch.manual_seed(17)
        recurrent = apply_cera(copy.deepcopy(base), rank=2, dropout=0,
                               act_fn="identity", target_modules=TARGETS,
                               variant="peft_aligned", alpha=2, recurrent_steps=4)
        self.assertTrue(torch.equal(rng_after_baseline, torch.get_rng_state()))
        for name, value in baseline.state_dict().items():
            if ".cera.A." in name or ".cera.B." in name:
                torch.testing.assert_close(value, recurrent.state_dict()[name], rtol=0, atol=0)
        recurrent_one = apply_cera(copy.deepcopy(base), rank=2, dropout=0,
                       act_fn="identity", target_modules=TARGETS,
                       variant="peft_aligned", alpha=2, recurrent_steps=1)
        for module in baseline.modules():
            if isinstance(module, CeRAWrapper):
                nn.init.normal_(module.cera.B.weight)
        baseline_state = baseline.state_dict()
        recurrent_state = recurrent.state_dict()
        with torch.no_grad():
            for name, value in baseline_state.items():
                if name in recurrent_state:
                    recurrent_state[name].copy_(value)
            recurrent_one_state = recurrent_one.state_dict()
            for name, value in recurrent_state.items():
                if name in recurrent_one_state:
                    recurrent_one_state[name].copy_(value)
        inputs = torch.randn(3, 28)
        baseline_layer = baseline.model.layers[0].mlp["down_proj"]
        recurrent_layer = recurrent.model.layers[0].mlp["down_proj"]
        recurrent_one_layer = recurrent_one.model.layers[0].mlp["down_proj"]
        torch.testing.assert_close(baseline_layer(inputs), recurrent_layer(inputs), rtol=0, atol=0)
        self.assertFalse(any("recurrent_" in name for name, _ in baseline.named_parameters()))
        recurrent_layer(inputs).sum().backward()
        recurrent_one_layer(inputs).sum().backward()
        self.assertGreater(recurrent_layer.cera.recurrent_outer.weight.grad.abs().sum().item(), 0)
        self.assertEqual(recurrent_layer.cera.recurrent_inner.weight.grad.abs().sum().item(), 0)
        torch.testing.assert_close(
            recurrent_layer.cera.recurrent_outer.weight.grad,
            recurrent_one_layer.cera.recurrent_outer.weight.grad,
        )

        with self.assertRaisesRegex(ValueError, "requires variant='peft_aligned'"):
            apply_cera(tiny_model(), rank=2, act_fn="identity",
                       target_modules=TARGETS, recurrent_steps=1)

    def test_recurrent_checkpoint_round_trip(self):
        base = tiny_model()
        model = apply_cera(copy.deepcopy(base), rank=2, dropout=0,
                           act_fn="identity", target_modules=TARGETS,
                           variant="peft_aligned", alpha=2, recurrent_steps=2)
        for module in model.modules():
            if isinstance(module, CeRAWrapper):
                nn.init.normal_(module.cera.B.weight)
                nn.init.normal_(module.cera.recurrent_outer.weight)
        config = dict(
            cera_format_version=3, rank_mode="fixed", model_type="CeRA",
            scale=1.0, alpha=2, rank=2, dropout=0.0, act_fn="identity",
            adapter_dtype="float32", target_modules=",".join(TARGETS),
            cera_variant="peft_aligned", recurrent_steps=2,
            recurrent_activation="silu", recurrent_step_weight_sharing=True,
            recurrent_update_scale=0.5,
        )
        directory = tempfile.mkdtemp(prefix="cera-recurrent-checkpoint-test-")
        save_checkpoint(model, 1, 64, {}, directory, "CeRA", config=config)
        checkpoint = torch.load(Path(directory) / "cera_ckpt_64.pt", weights_only=True)
        restored = restore_cera(copy.deepcopy(base), checkpoint, rank=2, dropout=0,
                                act_fn="identity", target_modules=TARGETS)
        model.eval()
        restored.eval()
        inputs = torch.randn(3, 28)
        torch.testing.assert_close(
            model.model.layers[0].mlp["down_proj"](inputs),
            restored.model.layers[0].mlp["down_proj"](inputs),
        )

    def test_learned_mix_checkpoint_round_trip_and_metadata(self):
        base = tiny_model()
        model = apply_cera(
            copy.deepcopy(base), rank=2, dropout=0, act_fn="silu",
            target_modules=TARGETS, variant="peft_aligned", alpha=2,
            mix_mode="learned_mix", gamma_init=0,
        )
        wrappers = [module for module in model.modules() if isinstance(module, CeRAWrapper)]
        with torch.no_grad():
            for index, module in enumerate(wrappers):
                nn.init.normal_(module.cera.B.weight)
                module.cera.gamma.fill_(index / 10)
        expected_budget = sum(
            2 * (module.original_layer.in_features + module.original_layer.out_features) + 1
            for module in wrappers
        )
        self.assertEqual(validate_cera_budget(model, 2, len(TARGETS)), expected_budget)
        config = dict(
            cera_format_version=4, rank_mode="fixed", model_type="CeRA",
            scale=1.0, alpha=2, rank=2, dropout=0.0, act_fn="silu",
            adapter_dtype="float32", target_modules=",".join(TARGETS),
            cera_variant="peft_aligned", recurrent_steps=0,
            cera_mix_mode="learned_mix", gamma_init=0.0,
            gamma_granularity="per_adapter_module",
            gamma_parameterization="unconstrained_scalar",
            mix_formula="linear_silu_interpolation",
        )
        directory = tempfile.mkdtemp(prefix="cera-mix-checkpoint-test-")
        save_checkpoint(model, 1, 64, {}, directory, "CeRA", config=config)
        checkpoint = torch.load(Path(directory) / "cera_ckpt_64.pt", weights_only=True)
        restored = restore_cera(
            copy.deepcopy(base), checkpoint, rank=2, dropout=0,
            act_fn="silu", target_modules=TARGETS,
        )
        model.eval()
        restored.eval()
        inputs = torch.randn(3, 28)
        torch.testing.assert_close(
            model.model.layers[0].mlp["down_proj"](inputs),
            restored.model.layers[0].mlp["down_proj"](inputs),
        )
        self.assertEqual(
            [module.cera.gamma.item() for module in wrappers],
            [module.cera.gamma.item() for module in restored.modules()
             if isinstance(module, CeRAWrapper)],
        )

        invalid_config = dict(config)
        invalid_config.pop("gamma_granularity")
        with self.assertRaisesRegex(ValueError, "learned-mix CeRA metadata"):
            validate_cera_config(invalid_config)

    def test_evaluation_loader_reconstructs_logits_and_rejects_conflict(self):
        import evaluate
        from transformers import LlamaConfig, LlamaForCausalLM

        base = LlamaForCausalLM(LlamaConfig(
            vocab_size=32, hidden_size=8, intermediate_size=28, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1,
        )).to(torch.bfloat16)
        base.requires_grad_(False)
        model = apply_cera(copy.deepcopy(base), rank=2, target_modules=TARGETS,
                           dropout=0.1, adapter_dtype=torch.float32)
        for module in model.modules():
            if isinstance(module, CeRAWrapper):
                nn.init.normal_(module.cera.B.weight, std=0.05)
        config = dict(cera_format_version=1, rank_mode="fixed", model_type="CeRA",
                      model="test/tiny", scale=1.0, rank=2, dropout=0.1, act_fn="silu",
                      adapter_dtype="float32", target_modules=",".join(TARGETS))
        directory = tempfile.mkdtemp(prefix="cera-eval-test-")
        save_checkpoint(model, 1, 64, {}, directory, "CeRA", config=config)
        args = argparse.Namespace(adapter_type="cera", adapter_format="legacy", rank=2,
                                  base_model="test/tiny", dropout=0.1, act_fn="silu",
                                  target_modules="q_proj,v_proj", explicit_options={"--rank"},
                                  checkpoint=str(Path(directory) / "cera_ckpt_64.pt"))
        with patch.object(evaluate.AutoModelForCausalLM, "from_pretrained", return_value=base), \
             patch.object(evaluate.AutoTokenizer, "from_pretrained", return_value=MagicMock()):
            restored, _ = evaluate.load_model_with_adapter(args, "unused")
        model.eval()
        inputs = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            torch.testing.assert_close(model(inputs).logits, restored(inputs).logits)
        args.rank = 3
        with self.assertRaisesRegex(ValueError, "conflicts"):
            evaluate.load_model_with_adapter(args, "unused")
        print(f"[TEST ARTIFACT] {directory}")

    def test_checkpoint_selection_is_exact_and_rejects_ambiguity(self):
        script = Path(__file__).resolve().parents[1] / "slurm/run_eval_peft.sh"
        source = script.read_text().split("<<'PYSELECT'\n", 1)[1].split("\nPYSELECT", 1)[0]
        directory = Path(tempfile.mkdtemp(prefix="cera-selection-test-"))
        config = dict(model_type="CeRA", dataset="metamathqa", rank=64, lr=1e-4,
                      dropout=0.1, model="test/tiny", seed=42, cera_format_version=1,
                      act_fn="silu", target_modules=",".join(TARGETS))
        command = [sys.executable, "-c", source, "CeRA", "metamathqa", "64", "1e-4", "0.1",
                   "test/tiny", "64", ",".join(TARGETS), "42", "silu", "legacy", "0",
                   "pure", "0.0"]
        for index in range(2):
            output = directory / "results" / f"Exp_PEFT_CeRA_test_{index}" / "CeRA"
            output.mkdir(parents=True)
            (output / "CeRA_log.json").write_text(json.dumps({"config": config}))
            (output / "cera_ckpt_best_1.pt").touch()
            result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
            if index == 0:
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("cera_ckpt_best_1.pt", result.stdout)
            else:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("found 2", result.stderr)
        print(f"[TEST ARTIFACT] {directory}")

    def test_checkpoint_selection_distinguishes_learned_mix(self):
        script = Path(__file__).resolve().parents[1] / "slurm/run_eval_peft.sh"
        source = script.read_text().split("<<'PYSELECT'\n", 1)[1].split("\nPYSELECT", 1)[0]
        directory = Path(tempfile.mkdtemp(prefix="cera-mix-selection-test-"))
        common = dict(
            model_type="CeRA", dataset="metamathqa", rank=64, lr=1e-4,
            dropout=0.0, model="test/tiny", seed=42, act_fn="silu", alpha=64,
            target_modules="q_proj,v_proj", cera_variant="peft_aligned",
            recurrent_steps=0,
        )
        for mix_mode in ("pure", "learned_mix"):
            output = directory / "results" / f"Exp_PEFT_CeRA_{mix_mode}" / "CeRA"
            output.mkdir(parents=True)
            config = dict(common, cera_mix_mode=mix_mode, gamma_init=0.0)
            (output / "CeRA_log.json").write_text(json.dumps({"config": config}))
            (output / "cera_ckpt_best_1.pt").touch()
        command = [
            sys.executable, "-c", source, "CeRA", "metamathqa", "64", "1e-4", "0.0",
            "test/tiny", "64", "q_proj,v_proj", "42", "silu", "peft_aligned", "0",
            "learned_mix", "0.0",
        ]
        result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Exp_PEFT_CeRA_learned_mix", result.stdout)

    def test_explicit_checkpoint_canonicalizes_mix_metadata(self):
        script = Path(__file__).resolve().parents[1] / "slurm/run_eval_peft.sh"
        source = script.read_text().split("<<'PYMETADATA'\n", 1)[1].split(
            "\nPYMETADATA", 1
        )[0]
        directory = Path(tempfile.mkdtemp(prefix="cera-mix-metadata-test-"))
        config = dict(
            cera_format_version=4, rank_mode="fixed", model_type="CeRA",
            model="test/checkpoint-model", scale=1.0, alpha=2, rank=2,
            dropout=0.0, act_fn="silu", adapter_dtype="float32",
            target_modules="q_proj,v_proj", cera_variant="peft_aligned",
            recurrent_steps=0, cera_mix_mode="learned_mix", gamma_init=0.0,
            gamma_granularity="per_adapter_module",
            gamma_parameterization="unconstrained_scalar",
            mix_formula="linear_silu_interpolation",
        )
        checkpoint_path = directory / "cera_ckpt_best_1.pt"
        torch.save({"config": config}, checkpoint_path)
        command = [
            sys.executable, "-c", source, str(checkpoint_path),
            "wrong/model", "64", "0.1", "all_linear", "identity", "legacy",
            "3", "pure", "1.0", "64",
        ]
        result = subprocess.run(command, cwd=script.parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip().split("\t"),
            [
                "test/checkpoint-model", "2", "0.0", "q_proj,v_proj", "silu",
                "peft_aligned", "0", "learned_mix", "0.0", "2",
            ],
        )

        pure_config = dict(config)
        pure_config.update(cera_format_version=2, act_fn="identity")
        for key in (
            "cera_mix_mode", "gamma_init", "gamma_granularity",
            "gamma_parameterization", "mix_formula",
        ):
            pure_config.pop(key, None)
        pure_checkpoint_path = directory / "cera_ckpt_best_2.pt"
        torch.save({"config": pure_config}, pure_checkpoint_path)
        pure_command = list(command)
        pure_command[3] = str(pure_checkpoint_path)
        pure_result = subprocess.run(
            pure_command, cwd=script.parents[1], capture_output=True, text=True
        )
        self.assertEqual(pure_result.returncode, 0, pure_result.stderr)
        pure_fields = pure_result.stdout.strip().split("\t")
        self.assertEqual(pure_fields[7:9], ["pure", "0.0"])


if __name__ == "__main__":
    unittest.main()