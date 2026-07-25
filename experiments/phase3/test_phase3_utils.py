#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from budget import BudgetManifest, record_training_step  # noqa: E402
from common import ABSORPTION_CONFIGS, MAIN_TO_ABSORPTION, absorption_config  # noqa: E402
from prefix_audit import PrefixCandidate, run_audit  # noqa: E402


class Phase3UtilsTest(unittest.TestCase):
    def test_u1_u4_same_non_expert_batch_budget(self):
        u1 = ABSORPTION_CONFIGS["U1"]
        u4 = ABSORPTION_CONFIGS["U4"]
        self.assertEqual(u1["rollout_per_batch"], u4["rollout_per_batch"] + u4["prefix_per_batch"])
        self.assertAlmostEqual(float(u1["lambda_rollout"]), 0.05)
        self.assertAlmostEqual(float(u4["lambda_rollout"]) + float(u4["lambda_prefix"]), 0.05)

    def test_main_maps_to_absorption(self):
        cfg = absorption_config("A4")
        self.assertEqual(cfg["absorption_id"], "U4")
        self.assertTrue(cfg["group_stratified"])

    def test_prefix_audit_pass(self):
        candidates = [
            PrefixCandidate(
                task="place_container_plate",
                env_seed=i,
                rollout_id=0,
                hdf5_path=f"path{i}.hdf5",
                total_steps=100,
                prefix_steps=70,
                prefix_fraction=0.7,
                extraction_mode="fixed_fraction",
            )
            for i in range(12)
        ]
        report = run_audit("place_container_plate", candidates, min_candidates=10)
        self.assertTrue(report.audit_passed)

    def test_budget_manifest_training_mass(self):
        manifest = BudgetManifest(
            task="t",
            main_id="A1",
            absorption_id="U1",
            train_seed=0,
        )
        record_training_step(
            manifest,
            expert_count=122,
            rollout_count=6,
            prefix_count=0,
            weighted_expert_contrib=1.0,
            weighted_rollout_contrib=0.05,
        )
        self.assertEqual(manifest.training.optimizer_steps, 1)
        self.assertAlmostEqual(manifest.training.loss_mass_non_expert, 0.05)


if __name__ == "__main__":
    unittest.main()
