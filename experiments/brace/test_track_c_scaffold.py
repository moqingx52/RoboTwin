#!/usr/bin/env python3
"""Unit tests for artifact inventory, validation, and confirmatory gate stats."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.brace.anchor_smoke import anchor_gate_passed
from experiments.brace.anchor_unit_smoke import run_unit_anchor_smoke
from experiments.brace.evaluate_confirmatory_gate import merge_task_points
from experiments.brace.inventory_artifacts import build_inventory, inspect_artifact, ArtifactSpec
from experiments.brace.inventory_traced_rollouts import inventory_task
from experiments.brace.merge_replay_audit_summaries import merge_summaries
from experiments.brace.orchestrate import prepare_resume_state, summarize_screen
from experiments.brace.run_eval_group import seed_shards_from_partial
from experiments.brace.promote_run import copy_file
from experiments.brace.replay_audit import write_json_atomic
from experiments.brace.resolve_artifact import resolve_audit_summary, resolve_branch_dir
from experiments.brace.screen_gates import compute_forgetting_gate, evaluate_constraint_feasibility, summarize_feasibility_trajectory
from experiments.brace.stage_records import emit_stage_record, list_records, resolve_latest_record
from experiments.brace.validate_artifacts import run_validation


class TrackCScaffoldTest(unittest.TestCase):
    def test_run_all_shell_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n", "experiments/brace/run_all.sh"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_constraint_feasibility_fails_on_missing_groups(self) -> None:
        protocol = {
            "anchor_smoke": {"identity_epsilon": 1e-4},
            "constraint_feasibility": {
                "mean_tolerance_factor": 2.0,
                "p90_max_factor": 5.0,
                "violation_fraction_max": {"base_solved": 0.5, "boundary": 0.5},
            },
        }
        summary = evaluate_constraint_feasibility([], protocol)
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["missing_groups"], ["base_solved", "boundary"])

    def test_preservation_group_batch_sampler_small_dataset(self) -> None:
        from experiments.brace.preservation_sampler import PreservationGroupBatchSampler

        groups = np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int64)
        sampler = PreservationGroupBatchSampler(
            groups,
            batch_size=4,
            preservation_group_ids={"base_solved": 1, "boundary": 2},
            samples_per_group=2,
            seed=0,
            num_batches=3,
        )
        batches = list(sampler)
        self.assertEqual(len(batches), 3)
        for batch in batches:
            self.assertEqual(len(batch), 4)
            labels = groups[batch]
            self.assertEqual(int(np.sum(labels == 1)), 2)
            self.assertEqual(int(np.sum(labels == 2)), 2)

    def test_forgetting_gate_rejects_joint_catastrophic_forgetting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_solved_seeds = {100, 101, 102, 104, 105}

            def write_eval(path: Path, outcomes: dict[int, bool]) -> None:
                payload = {
                    "rows": [
                        {"env_seed": seed, "split": "train_seen", "success": outcome}
                        for seed, outcome in outcomes.items()
                    ]
                }
                path.write_text(json.dumps(payload), encoding="utf-8")

            base_path = root / "base.json"
            u1_path = root / "u1.json"
            b2_path = root / "b2.json"
            outcomes = {seed: True for seed in base_solved_seeds}
            write_eval(base_path, outcomes)
            write_eval(u1_path, {seed: False for seed in base_solved_seeds})
            write_eval(b2_path, {seed: False for seed in base_solved_seeds})
            protocol = {
                "preservation_metrics": {
                    "min_paired_seeds": 5,
                    "require_b2_forgetting_below_u1": True,
                }
            }
            summary = compute_forgetting_gate(
                base_eval=base_path,
                u1_eval=u1_path,
                b2_eval=b2_path,
                base_solved_seeds=base_solved_seeds,
                protocol=protocol,
            )
            self.assertFalse(summary["passed"])
            self.assertAlmostEqual(summary["forgetting_rate_u1"], 1.0)
            self.assertAlmostEqual(summary["forgetting_rate_b2"], 1.0)

    def test_forgetting_gate_prefers_b2_when_u1_forgets_more(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_solved_seeds = {100, 101, 102, 103, 104}

            def write_eval(path: Path, outcomes: dict[int, bool]) -> None:
                payload = {
                    "rows": [
                        {"env_seed": seed, "split": "train_seen", "success": outcome}
                        for seed, outcome in outcomes.items()
                    ]
                }
                path.write_text(json.dumps(payload), encoding="utf-8")

            base_path = root / "base.json"
            u1_path = root / "u1.json"
            b2_path = root / "b2.json"
            write_eval(base_path, {seed: True for seed in base_solved_seeds})
            write_eval(
                u1_path,
                {100: True, 101: True, 102: False, 103: False, 104: False},
            )
            write_eval(b2_path, {seed: True for seed in base_solved_seeds})
            protocol = {
                "preservation_metrics": {
                    "min_paired_seeds": 5,
                    "require_b2_forgetting_below_u1": True,
                }
            }
            summary = compute_forgetting_gate(
                base_eval=base_path,
                u1_eval=u1_path,
                b2_eval=b2_path,
                base_solved_seeds=base_solved_seeds,
                protocol=protocol,
            )
            self.assertTrue(summary["passed"])
            self.assertGreater(summary["forgetting_rate_u1"], summary["forgetting_rate_b2"])

    def test_summarize_feasibility_trajectory_reports_tail_window(self) -> None:
        rows = []
        for step in range(10):
            rows.append(
                {
                    "step": step,
                    "brace_constraint/base_solved": 0.02 if step < 8 else 0.001,
                    "brace_constraint/boundary": 0.02 if step < 8 else 0.001,
                    "brace_monitor/base_solved_ema_drift": 0.015 if step < 8 else 0.001,
                    "brace_dual/base_solved": 0.01 * step,
                }
            )
        summary = summarize_feasibility_trajectory(rows, epsilon=1e-4, tail_fraction=0.2)
        self.assertEqual(summary["rows"], 10)
        self.assertGreater(summary["base_solved"]["tail_mean"], 0.0)
        self.assertLess(summary["base_solved"]["tail_mean"], summary["base_solved"]["full_mean"])

    def test_anchor_smoke_passes(self) -> None:
        protocol = json.loads(
            Path("experiments/brace/screen_protocol.v1.json").read_text(encoding="utf-8")
        )
        summary = run_unit_anchor_smoke(protocol)
        self.assertTrue(summary["passed"])
        self.assertFalse(summary["eligible_for_screen_gate"])
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

    def test_resolve_audit_summary_skips_failed_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            brace_dir = Path(tmp)
            runs = brace_dir / "runs"
            archive = brace_dir / "archive" / "replay_audit_v2_place_v2.3_gate"
            failed_run = brace_dir / "failed_run"
            runs.mkdir(parents=True)
            failed_run.mkdir(parents=True)
            archive.mkdir(parents=True)
            failed_summary = {
                "passed": False,
                "tasks": {"place_container_plate": {"replay_gate_passed": False}},
            }
            passed_summary = {
                "passed": True,
                "tasks": {"place_container_plate": {"replay_gate_passed": True}},
            }
            write_json_atomic(failed_run / "summary.json", failed_summary)
            write_json_atomic(archive / "summary.json", passed_summary)
            (runs / "LATEST_AUDIT_place_container_plate").write_text(
                str(failed_run), encoding="utf-8"
            )
            path = resolve_audit_summary("place_container_plate", brace_dir=brace_dir)
            self.assertIsNotNone(path)
            self.assertEqual(path, archive / "summary.json")

    def test_resolve_audit_summary_explicit_override_must_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            brace_dir = Path(tmp)
            failed_dir = brace_dir / "explicit_failed"
            failed_dir.mkdir(parents=True)
            write_json_atomic(
                failed_dir / "summary.json",
                {
                    "passed": False,
                    "tasks": {"place_container_plate": {"replay_gate_passed": False}},
                },
            )
            os.environ["BRACE_AUDIT_RUN_DIR"] = str(failed_dir)
            try:
                with self.assertRaises(ValueError):
                    resolve_audit_summary("place_container_plate", brace_dir=brace_dir)
            finally:
                os.environ.pop("BRACE_AUDIT_RUN_DIR", None)

    def test_resolve_audit_summary_passed_latest_preferred(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            brace_dir = Path(tmp)
            runs = brace_dir / "runs"
            archive = brace_dir / "archive" / "replay_audit_v2_place_v2.3_gate"
            passed_run = brace_dir / "passed_run"
            runs.mkdir(parents=True)
            passed_run.mkdir(parents=True)
            archive.mkdir(parents=True)
            passed_summary = {
                "passed": True,
                "tasks": {"place_container_plate": {"replay_gate_passed": True}},
            }
            write_json_atomic(passed_run / "summary.json", passed_summary)
            write_json_atomic(archive / "summary.json", passed_summary)
            (runs / "LATEST_AUDIT_place_container_plate").write_text(
                str(passed_run), encoding="utf-8"
            )
            path = resolve_audit_summary("place_container_plate", brace_dir=brace_dir)
            self.assertEqual(path, passed_run / "summary.json")

    def test_anchor_gate_requires_training_path(self) -> None:
        protocol_revision = "screen.v1.1"
        self.assertFalse(
            anchor_gate_passed(
                {"passed": True, "gate_level": "unit", "protocol_revision": protocol_revision},
                protocol_revision,
            )
        )
        self.assertTrue(
            anchor_gate_passed(
                {
                    "passed": True,
                    "gate_level": "training_path",
                    "protocol_revision": protocol_revision,
                    "checks": {
                        "unit_smoke_passed": True,
                        "training_path_smoke_passed": True,
                    },
                },
                protocol_revision,
            )
        )
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

    def test_development_screen_summary_applies_v12_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = {}
            feasibility = {
                "passed": True,
                "missing_groups": [],
                "groups": {
                    "base_solved": {"passed": True},
                    "boundary": {"passed": True},
                },
            }
            base_solved_seeds = [100, 101, 102, 103, 104]

            def write_eval(job_id, method, epoch, values, seed_outcomes=None):
                path = root / f"{method}_{epoch}.json"
                payload = {
                    "splits": {
                        split: {"mean_sr": value}
                        for split, value in zip(("id_heldout", "train_seen", "hard_20"), values)
                    }
                }
                if seed_outcomes is not None:
                    payload["rows"] = [
                        {"env_seed": seed, "split": "train_seen", "success": outcome}
                        for seed, outcome in seed_outcomes.items()
                    ]
                path.write_text(json.dumps(payload), encoding="utf-8")
                jobs[job_id] = {"artifact": str(path)}

            def write_train(job_id, method, epoch):
                ckpt = root / f"{method}_{epoch}.ckpt"
                ckpt.write_bytes(b"ckpt")
                feas_path = ckpt.with_suffix(".feasibility.json")
                feas_path.write_text(json.dumps(feasibility), encoding="utf-8")
                jobs[job_id] = {"artifact": str(ckpt)}

            base_outcomes = {seed: True for seed in base_solved_seeds}
            write_eval("eval:base", "base", 0, (0.5, 0.5, 0.5), base_outcomes)
            u1_outcomes = {seed: True for seed in base_solved_seeds}
            u1_outcomes[104] = False
            scores = {
                "N1": (0.45, 0.40, 0.20),
                "B1": (0.46, 0.41, 0.30),
                "B2": (0.47, 0.43, 0.35),
                "B3": (0.49, 0.46, 0.40),
            }
            for method, values in scores.items():
                outcomes = u1_outcomes if method == "N1" else base_outcomes
                write_eval(f"eval:{method}:epoch1", method, 1, values, outcomes)
                if method in {"B2", "B3"}:
                    write_train(f"train:{method}:epoch1", method, 1)
            anchor_manifest_dir = root / "datasets" / "place_container_plate_anchor_replay.zarr"
            anchor_manifest_dir.mkdir(parents=True)
            anchor_manifest = {
                "episodes": [
                    {"env_seed": seed, "preservation_group": "base_solved"} for seed in base_solved_seeds
                ]
            }
            (anchor_manifest_dir / "brace_anchor_manifest.json").write_text(
                json.dumps(anchor_manifest), encoding="utf-8"
            )
            state = {
                "task": "place_container_plate",
                "run_label": "test",
                "run_dir": str(root),
                "screen_epochs": [1],
                "active_methods": ["N1", "B1", "B2", "B3"],
                "jobs": jobs,
            }
            protocol = {
                "screen_mode": "full",
                "screen_epochs": [1],
                "promotion": {"id_fraction_of_base": 0.8, "train_fraction_of_base": 0.75},
                "eval": {"hard_for_checkpoint_selection": False},
                "screens": {"credit": {"min_score_margin": 0.01}},
                "preservation_metrics": {
                    "base_solved_forgetting_max_delta": 0.05,
                    "min_paired_seeds": 5,
                    "require_b2_forgetting_below_u1": True,
                },
            }
            summary = summarize_screen(state, protocol)
            self.assertTrue(summary["passed"])
            self.assertTrue(summary["screens"]["preservation_U1_vs_B2"]["passed"])
            self.assertTrue(summary["screens"]["credit_B1_vs_N1"]["passed"])
            self.assertEqual(summary["screens"]["integration_B1_B2_B3"]["best_method"], "B3")
            self.assertTrue(summary["best_eligible"]["B2"]["constraint_feasibility"]["passed"])
            self.assertTrue(summary["best_eligible"]["B2"]["forgetting_gate"]["passed"])
            self.assertAlmostEqual(summary["methods"]["B1"]["1"]["selection_score"], 0.82, places=5)

    def test_validate_restored_branch_artifacts(self) -> None:
        result = run_validation(
            inventory_path=None,
            missing_only=False,
            strict_json=False,
            run_label="place_pilot_v2.3",
        )
        self.assertTrue(result["passed"], result)

    def test_orchestrate_launch_pins_cuda_visible_devices(self) -> None:
        from unittest.mock import MagicMock, patch

        from experiments.brace.orchestrate import Runner

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "job.log"
            state = {
                "jobs": {
                    "eval:test": {
                        "id": "eval:test",
                        "status": "pending",
                        "command": ["python", "-c", "print(0)"],
                        "log": str(log_path),
                        "attempts": 0,
                    }
                },
                "events": [],
            }
            args = MagicMock()
            args.max_retries = 1
            runner = Runner(args, Path(tmp) / "state.json", state, {})
            with patch("experiments.brace.orchestrate.subprocess.Popen") as popen:
                popen.return_value = MagicMock(pid=123, poll=MagicMock(return_value=None))
                runner.launch(3, state["jobs"]["eval:test"])
            _, kwargs = popen.call_args
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "3")

    def test_prepare_resume_state_resets_failed_jobs(self) -> None:
        state = {
            "status": "failed",
            "jobs": {
                "eval:base": {"status": "completed", "attempts": 1},
                "eval:N1:epoch1": {"status": "failed", "attempts": 2, "exit_code": 1, "gpu": 0},
            },
        }
        prepare_resume_state(state)
        self.assertEqual(state["status"], "running")
        self.assertEqual(state["jobs"]["eval:base"]["status"], "completed")
        self.assertEqual(state["jobs"]["eval:N1:epoch1"]["status"], "pending")
        self.assertEqual(state["jobs"]["eval:N1:epoch1"]["attempts"], 0)
        self.assertNotIn("exit_code", state["jobs"]["eval:N1:epoch1"])

    def test_partial_eval_rows_are_migrated_to_resume_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seeds = root / "seeds.json"
            hard = root / "hard.json"
            output = root / "eval"
            task_dir = output / "place_container_plate"
            task_dir.mkdir(parents=True)
            seeds.write_text(
                json.dumps({"eval_id": [10, 11], "train_rollout": [20, 21]}),
                encoding="utf-8",
            )
            hard.write_text(json.dumps({"hard_seeds": [30, 31]}), encoding="utf-8")
            rows = [
                {"split": "id_heldout", "env_seed": 10, "repeat": 0, "policy_seed": 2000, "success": True},
                {"split": "train_seen", "env_seed": 20, "repeat": 0, "policy_seed": 2000, "success": False},
            ]
            parent = {
                "task_name": "place_container_plate",
                "task_config": "demo_clean",
                "variant": "B1_epoch1",
                "ckpt_path": "checkpoint.ckpt",
                "hard_seeds": [30, 31],
                "hard_seed_source": str(hard),
                "rows": rows,
                "splits": {},
                "progress": {"complete": False, "completed_episodes": 2},
            }
            (task_dir / "B1_epoch1.json").write_text(json.dumps(parent), encoding="utf-8")
            command = [
                "python", "eval_per_seed.py",
                "--task", "place_container_plate",
                "--task-config", "demo_clean",
                "--variant", "B1_epoch1",
                "--ckpt-path", "checkpoint.ckpt",
                "--output-dir", str(output),
                "--seeds-file", str(seeds),
                "--hard-seeds-file", str(hard),
                "--id-seed-count", "2",
                "--train-seed-count", "2",
                "--hard-seed-count", "2",
                "--id-repeats", "1",
                "--train-repeats", "1",
                "--hard-repeats", "1",
                "--policy-seed-offset", "2000",
                "--resume",
            ]
            self.assertEqual(seed_shards_from_partial(command, 3), 2)
            migrated_rows = []
            for shard in range(3):
                path = task_dir / f"B1_epoch1_shard_{shard:02d}_of_03.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(payload["progress"]["num_shards"], 3)
                self.assertEqual(payload["progress"]["shard_id"], shard)
                migrated_rows.extend(payload["rows"])
            self.assertCountEqual(migrated_rows, rows)


if __name__ == "__main__":
    unittest.main()
