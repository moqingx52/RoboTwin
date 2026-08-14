"""Focused tests for the low-memory BRACE gradient diagnostic helpers."""

from __future__ import annotations

import math
import unittest

import torch

from experiments.brace_v2_legacy.anchor_grad_diagnostics import (
    backward_accumulate_and_measure,
    flatten_grad_norm,
    measure_checkpoint_grad_metrics,
)


class AnchorGradDiagnosticsTest(unittest.TestCase):
    def test_sequential_backward_matches_summed_loss_gradient(self) -> None:
        param = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        raw = (param.square()).sum()
        anchor = ((param - 3.0).square()).sum() * 0.25
        expected = torch.autograd.grad(raw + anchor, param, retain_graph=True)[0]
        expected_sft = torch.autograd.grad(raw, param, retain_graph=True)[0]
        expected_anchor = torch.autograd.grad(anchor, param, retain_graph=True)[0]

        raw.backward()
        self.assertAlmostEqual(flatten_grad_norm([param]), float(expected_sft.norm()), places=6)
        anchor_norm, cosine = backward_accumulate_and_measure([param], anchor)

        torch.testing.assert_close(param.grad, expected)
        self.assertAlmostEqual(anchor_norm, float(expected_anchor.norm()), places=6)
        expected_cosine = float(
            torch.dot(expected_sft, expected_anchor) / (expected_sft.norm() * expected_anchor.norm())
        )
        self.assertAlmostEqual(float(cosine), expected_cosine, places=6)

    def test_checkpoint_metrics_reuse_sft_grads_without_sft_graph(self) -> None:
        param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        raw = (param.square()).sum()
        raw.backward()
        sft_norm = flatten_grad_norm([param])
        del raw

        constraint_a = (param[0] - 4.0).square()
        constraint_b = (param[1] + 1.0).square()
        scaled_anchor = 0.5 * constraint_a + 0.25 * constraint_b
        expected_anchor = torch.autograd.grad(scaled_anchor, param, retain_graph=True)[0]

        metrics = measure_checkpoint_grad_metrics(
            [param],
            scaled_anchor=scaled_anchor,
            anchor_constraints={"a": constraint_a, "b": constraint_b},
            grad_accum=1,
            dual_values={"a": 0.5, "b": 0.25},
            grad_norm_sft=sft_norm,
        )

        self.assertAlmostEqual(float(metrics["grad_norm_anchor"]), float(expected_anchor.norm()), places=6)
        self.assertTrue(math.isfinite(float(metrics["grad_cosine_sft_anchor"])))
        self.assertAlmostEqual(float(metrics["grad_norm_constraint/a"]), 6.0, places=6)
        self.assertAlmostEqual(float(metrics["grad_norm_constraint/b"]), 6.0, places=6)
        self.assertIsNone(param.grad)


if __name__ == "__main__":
    unittest.main()
