#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from promotion import evaluate_diagnose, evaluate_screen  # noqa: E402


def _eval_payload(id_sr, train_sr, id_cov=0.5, train_cov=0.5):
    return {
        "splits": {
            "id_heldout": {"mean_sr": id_sr, "solved_coverage": id_cov},
            "train_seen": {"mean_sr": train_sr, "solved_coverage": train_cov},
            "hard_20": {"mean_sr": 0.0, "solved_coverage": 0.0},
        }
    }


def source_separated_loss(expert_values, rollout_values, lambda_expert=0.9, lambda_rollout=0.1):
    expert_loss = sum(expert_values) / len(expert_values)
    rollout_loss = sum(rollout_values) / len(rollout_values)
    return lambda_expert * expert_loss + lambda_rollout * rollout_loss


class Phase2bUtilsTest(unittest.TestCase):
    def test_screen_promotion_pass(self):
        base = {"id_mean_sr": 0.5, "id_coverage": 0.5, "train_mean_sr": 0.5, "train_coverage": 0.5}
        result = evaluate_screen(
            "place_container_plate",
            _eval_payload(0.45, 0.40),
            base,
        )
        self.assertTrue(result["passed"])

    def test_screen_promotion_fail(self):
        base = {"id_mean_sr": 0.5, "id_coverage": 0.5, "train_mean_sr": 0.5, "train_coverage": 0.5}
        result = evaluate_screen(
            "place_container_plate",
            _eval_payload(0.10, 0.40),
            base,
        )
        self.assertFalse(result["passed"])

    def test_diagnose_normalizer_primary(self):
        summary = evaluate_diagnose(
            "place_container_plate",
            _eval_payload(0.5, 0.5),
            _eval_payload(0.30, 0.5),
        )
        self.assertTrue(summary["normalizer_is_primary_cause"])

    def test_source_separated_loss_math(self):
        loss = source_separated_loss([1.0, 2.0], [3.0, 4.0], 0.9, 0.1)
        self.assertAlmostEqual(loss, 0.9 * 1.5 + 0.1 * 3.5)


if __name__ == "__main__":
    unittest.main()
