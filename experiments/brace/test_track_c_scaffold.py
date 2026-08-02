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
from experiments.brace.orchestrate import summarize_screen
from experiments.brace.promote_run import copy_file
from experiments.brace.replay_audit import write_json_atomic
from experiments.brace.resolve_artifact import resolve_audit_summary, resolve_branch_dir
from experiments.brace.stage_records import emit_stage_record, list_records, resolve_latest_record
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
        )
        entry = inspect_artifact(spec)
        self.assertTrue(entry["exists"])
        self.assertEqual(entry["status"], "present")

    def test_resolve_audit_summary_archive_fallback(self) -> None:
        path = resolve_audit_summary("dump_bin_bigbin")
        self.assertIsNotNone(path)
        self.assertTrue(path.is_file())

    def test_resolve_branch_dir_archive_fallback(self) -> None:
        path = resolve_branch_dir("branches")
        self.assertIsNotNone(path)
        self.assertTrue((path / "summary.json").is_file())

    def test_stage_record_emit_and_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records_dir = Path(tmp) / "records"
            summary_path = Path(tmp) / "summary.json"
            write_json_atomic(
                summary_path,
                {"passed": True, "complete": True, "tasks": {"place_container_plate": {"passed": True}}},
            )
            import experiments.brace.stage_records as stage_records

            original = stage_records.RECORDS_DIR
            stage_records.RECORDS_DIR = records_dir
            try:
                record_path = emit_stage_record(
                    "branch",
                    summary_path=summary_path,
                    tasks=["place_container_plate"],
                    label="branches",
                )
                self.assertTrue(record_path.is_file())
                record = json.loads(record_path.read_text(encoding="utf-8"))
                bundle = records_dir / "bundles" / record["record_id"] / "summary.json"
                self.assertTrue(bundle.is_file())
                self.assertFalse(record["evidence_bundle"]["binary_artifacts_included"])
                latest = resolve_latest_record("branch", task="place_container_plate")
                self.assertIsNotNone(latest)
                rows = list_records(limit=5)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["stage"], "branch")
            finally:
                stage_records.RECORDS_DIR = original

    def test_frozen_promotion_refuses_different_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            destination = root / "archive" / "summary.json"
            source.write_text('{"value": 1}\n', encoding="utf-8")
            copy_file(source, destination)
            copy_file(source, destination)  # idempotent
            source.write_text('{"value": 2}\n', encoding="utf-8")
            with self.assertRaises(FileExistsError):
                copy_file(source, destination)

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

    def test_development_screen_summary_uses_all_three_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = {}

            def write_eval(job_id, method, epoch, values):
                path = root / f"{method}_{epoch}.json"
                payload = {
                    "splits": {
                        split: {"mean_sr": value}
                        for split, value in zip(("id_heldout", "train_seen", "hard_20"), values)
                    }
                }
                path.write_text(json.dumps(payload), encoding="utf-8")
                jobs[job_id] = {"artifact": str(path)}

            write_eval("eval:base", "base", 0, (0.5, 0.5, 0.5))
            scores = {
                "N1": (0.45, 0.40, 0.20),
                "B1": (0.45, 0.40, 0.30),
                "B2": (0.46, 0.42, 0.35),
                "B3": (0.48, 0.45, 0.40),
            }
            for method, values in scores.items():
                write_eval(f"eval:{method}:epoch1", method, 1, values)
            state = {
                "task": "place_container_plate",
                "run_label": "test",
                "jobs": jobs,
            }
            protocol = {
                "screen_epochs": [1],
                "promotion": {"id_fraction_of_base": 0.8, "train_fraction_of_base": 0.75},
            }
            summary = summarize_screen(state, protocol)
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["screens"]["integration_B1_B2_B3"]["best_method"], "B3")
            self.assertAlmostEqual(summary["methods"]["B1"]["1"]["selection_score"], 0.6)

    def test_validate_restored_branch_artifacts(self) -> None:
        result = run_validation(
            inventory_path=None,
            missing_only=False,
            strict_json=False,
            run_label="place_pilot_v2.3",
        )
        self.assertTrue(result["passed"], result)


if __name__ == "__main__":
    unittest.main()
