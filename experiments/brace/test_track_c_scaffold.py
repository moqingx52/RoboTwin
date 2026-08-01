#!/usr/bin/env python3
"""Unit tests for Track C scaffold and confirmatory gate helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.brace.anchor_smoke import run_anchor_smoke
from experiments.brace.evaluate_confirmatory_gate import merge_task_points
from experiments.brace.inventory_traced_rollouts import inventory_task
from experiments.brace.replay_audit import write_json_atomic


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

    def test_confirmatory_gate_merge(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
