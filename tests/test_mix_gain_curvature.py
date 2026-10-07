import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from analyze_mix_gain_curvature import (
    CONDITIONS,
    FIT_LEVELS,
    InterventionController,
    Moments,
    fit_linear_pieces,
    piece_energies,
    substitute_correction,
)
from cera.adapters import CeRAWrapper


def make_moments(z, c):
    moments = Moments(z.shape[1])
    moments.update(z, c)
    return moments


class FitTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.rank = 6
        self.z = torch.randn(4000, self.rank, dtype=torch.float64) * 0.9 + 0.1
        self.gamma = -0.02
        self.c = self.gamma * (F.silu(self.z) - self.z)

    def test_exact_affine_correction_is_recovered(self):
        C = torch.randn(self.rank, self.rank, dtype=torch.float64) * 0.01
        b = torch.randn(self.rank, dtype=torch.float64) * 0.01
        fits = fit_linear_pieces(make_moments(self.z, self.z @ C.T + b), ridge=0.0)
        self.assertTrue(torch.allclose(fits["matrix"]["C"], C, atol=1e-9))
        self.assertTrue(torch.allclose(fits["matrix"]["b"], b, atol=1e-9))

    def test_energies_match_direct_computation_and_levels_nest(self):
        moments = make_moments(self.z, self.c)
        fits = fit_linear_pieces(moments, ridge=0.0)
        residuals = []
        for level in FIT_LEVELS:
            C, b = fits[level]["C"], fits[level]["b"]
            direct = (self.c - self.z @ C.T - b).square().sum().item()
            energy = piece_energies(moments, C, b)
            self.assertAlmostEqual(energy["residual"], direct, delta=1e-9 * moments.cc + 1e-15)
            residuals.append(direct)
        self.assertGreaterEqual(residuals[0], residuals[1])
        self.assertGreaterEqual(residuals[1], residuals[2])
        # SiLU correction keeps a real nonlinear residual after the best scalar gain.
        self.assertGreater(residuals[0] / moments.cc, 0.05)

    def test_pieces_sum_to_full_correction(self):
        fits = fit_linear_pieces(make_moments(self.z, self.c))
        gamma = torch.tensor(self.gamma, dtype=torch.float64)
        full = substitute_correction("full", self.z, gamma, fits)
        self.assertTrue(torch.allclose(full, self.c))
        self.assertEqual(float(substitute_correction("zero", self.z, gamma, fits).abs().sum()), 0.0)
        scalar = (
            substitute_correction("gain_scalar", self.z, gamma, fits)
            + substitute_correction("offset_scalar", self.z, gamma, fits)
        )
        self.assertTrue(torch.allclose(
            scalar, substitute_correction("affine_scalar", self.z, gamma, fits)
        ))
        for level in FIT_LEVELS:
            total = (
                substitute_correction(f"affine_{level}", self.z, gamma, fits)
                + substitute_correction(f"curvature_{level}", self.z, gamma, fits)
            )
            self.assertTrue(torch.allclose(total, full))


class HookTests(unittest.TestCase):
    def test_full_matches_native_and_zero_removes_correction(self):
        torch.manual_seed(1)

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = CeRAWrapper(
                    nn.Linear(8, 8, bias=False), 8, 8, 0.5, dropout=0.0, act_fn="silu",
                    rank=4, variant="peft_aligned", scaling=1.0, mix_mode="learned_mix",
                )

        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Layer()])

            def forward(self, x):
                return self.layers[0].q_proj(x)

        model = Toy().eval()
        adapter = model.layers[0].q_proj.cera
        with torch.no_grad():
            adapter.B.weight.normal_()
            adapter.gamma.fill_(-0.3)
        x = torch.randn(3, 5, 8)
        with torch.no_grad():
            native = model(x)
            controller = InterventionController(model)
            name = next(iter(controller.adapters))
            moments = controller.new_moments()
            controller.mask = torch.ones(3, 5, dtype=torch.bool)
            controller.collect = moments
            self.assertTrue(torch.allclose(model(x), native, atol=1e-6))
            controller.collect = None
            self.assertEqual(moments[name].n, 15)
            controller.set_fits({name: fit_linear_pieces(moments[name])})

            outputs = {}
            for condition in CONDITIONS:
                controller.condition = condition
                outputs[condition] = model(x)
            base = model.layers[0].q_proj.original_layer(x)
            linear = base + adapter.B(adapter.A(x)) * adapter.scaling
            self.assertTrue(torch.allclose(outputs["zero"], linear, atol=1e-6))
            self.assertFalse(torch.allclose(outputs["full"], linear, atol=1e-4))
            # B is linear, so affine + curvature deltas recombine into the full delta.
            for level in FIT_LEVELS:
                recombined = (
                    outputs[f"affine_{level}"] + outputs[f"curvature_{level}"] - linear
                )
                self.assertTrue(torch.allclose(recombined, outputs["full"], atol=1e-5))
            controller.close()
            self.assertTrue(torch.allclose(model(x), native))


if __name__ == "__main__":
    unittest.main()
