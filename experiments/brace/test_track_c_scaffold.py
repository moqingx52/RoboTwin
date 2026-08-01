#!/usr/bin/env python3
"""Unit tests for artifact inventory, validation, and confirmatory gate stats."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.brace.anchor_smoke import run_anchor_smoke
from experiments.brace.evaluate_confirmatory_gate import merge_task_points
from experiments.brace.inventory_artifacts import build_inventory, inspect_artifact, ArtifactSpec
from experiments.brace.inventory_traced_rollouts import inventory_task
from experiments.brace.merge_replay_audit_summaries import merge_summaries
from experiments.brace.replay_audit import write_json_atomic
from experiments.brace.validate_artifacts import run_validation


class TrackCScaffoldTest(unittest.TestCase):
    def test_anchor_smoke_passes(self) -> None:
        protocol = json.loads(
            Path("experiments/brace/screen_protocol.v1.json").read_text(encoding="utf-8")
        )
        summary = run_anchor_smoke(protocol)
        self.assertTrue(summary["passed"])
        self.assertTrue(summary["checks"]["dual_nonnegativity"])

    def test_inventory_mixed_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp)
            manifest = task_dir / "manifest_shard_00_of_01.jsonl"
            rows = [
                {"env_seed": 1, "rollout_id": 0, "success": True},
                {"env_seed": 1, "rollout_id": 1, "success": False},
                {"env_seed": 2, "rollout_id": 0, "success": True},
            ]
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            payload = inventory_task(task_dir)
            self.assertEqual(payload["mixed_outcome_count"], 1)
            self.assertEqual(payload["mixed_outcome_seeds"], [1])

    def test_confirmatory_gate_merge_seed_level(self) -> None:
        point = {
            "accepted": True,
            "candidate_success_rate": 1.0,
            "control_success_rate": 0.0,
            "env_seed": 100,
            "point_type": "first_persistent_divergence",
        }
        pilot = {
            "harness_valid": True,
            "tasks": {
                "place_container_plate": {
                    "points": [dict(point, env_seed=100 + i) for i in range(15)]
                }
            },
        }
        confirm = {
            "harness_valid": True,
            "tasks": {
                "place_container_plate": {
                    "points": [dict(point, env_seed=200 + i) for i in range(15)]
                }
            },
        }
        merged = merge_task_points(pilot, confirm, "place_container_plate")
        self.assertTrue(merged["checks"]["min_seeds"])
        self.assertTrue(merged["checks"]["min_points"])
        self.assertTrue(merged["checks"]["lift_positive"])
        self.assertAlmostEqual(merged["seed_level_mean_lift"], 1.0)
        self.assertIn("seed_level_bootstrap_ci", merged)
        self.assertIn("candidate_positive_rate_by_seed", merged)

    def test_inventory_detects_present_summary(self) -> None:
        spec = ArtifactSpec(
            "place_pilot_archive",
            "experiments/brace/archive/branches_place_pilot_valid_v2.3/summary.json",
            ("experiments/brace/archive/branches_place_pilot_valid_v2.3/summary.json",),
        )
        entry = inspect_artifact(spec)
        self.assertTrue(entry["exists"])
        self.assertEqual(entry["status"], "present")

    def test_build_inventory_has_bundle_summary(self) -> None:
        inventory = build_inventory()
        self.assertIn("bundle_summary", inventory)
        self.assertIn("artifacts", inventory)

    def test_merge_replay_audit_summaries(self) -> None:
        dump_summary = json.loads(
            Path("experiments/brace/archive/replay_audit_v2_dump_v2.3_gate/summary.json").read_text(
                encoding="utf-8"
            )
        )
        merged = merge_summaries([dump_summary])
        self.assertTrue(merged["tasks"]["dump_bin_bigbin"]["replay_gate_passed"])

    def test_validate_artifacts_warns_on_missing_checks(self) -> None:
        result = run_validation(
            inventory_path=None,
            missing_only=False,
            strict_json=False,
            run_label="place_pilot_v2.3",
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any("checks.jsonl" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
