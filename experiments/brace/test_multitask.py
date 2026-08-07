#!/usr/bin/env python3
"""Tests for the BRACE RoboTwin multitask preregistration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments.brace.aggregate_multitask import (
    aggregate_multitask,
    effect_maps_for_task,
    hierarchical_task_bootstrap,
    load_artifact_index,
    rows_by_key,
    validate_eval_payload,
)
from experiments.brace.multitask_protocol import (
    build_seed_manifest,
    file_sha256,
    validate_method_pilot_summary,
    validate_multitask_protocol,
)
from experiments.brace.multitask_scheduler import Scheduler, validate_job_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"


def eval_payload(success_by_split: dict[str, list[bool]]) -> dict:
    rows = []
    for split, outcomes in success_by_split.items():
        for env_seed, success in enumerate(outcomes, start=100):
            rows.append({"split": split, "env_seed": env_seed, "repeat": 0, "success": success})
    return {"progress": {"complete": True}, "rows": rows}


class MultitaskProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol_path = BRACE_DIR / "multitask_protocol.v1.json"
        self.protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        task_path = REPO_ROOT / self.protocol["task_manifest"]
        self.tasks = json.loads(task_path.read_text(encoding="utf-8"))

    def test_design_protocol_and_public_snapshot_validate(self) -> None:
        result = validate_multitask_protocol(self.protocol_path)
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(len(result["development_tasks"]), 2)
        self.assertEqual(len(result["heldout_tasks"]), 10)

    def test_design_assets_have_valid_sha256_sidecars(self) -> None:
        from experiments.brace.multitask_protocol import sha256_sidecar_valid

        for name in (
            "multitask_protocol.v1.json",
            "multitask_tasks.v1.json",
            "leaderboard_snapshot_20260807.json",
        ):
            self.assertTrue(sha256_sidecar_valid(BRACE_DIR / name), name)

    def test_heldout_launch_fails_without_method_freeze(self) -> None:
        result = validate_multitask_protocol(self.protocol_path, require_method_freeze=True)
        self.assertFalse(result["passed"])
        self.assertTrue(any("method freeze" in error for error in result["errors"]))

    def test_seed_partitions_are_disjoint_and_task_specific(self) -> None:
        first = build_seed_manifest(self.protocol, self.tasks, "place_container_plate")
        second = build_seed_manifest(self.protocol, self.tasks, "dump_bin_bigbin")
        first_values = [seed for values in first["partitions"].values() for seed in values]
        second_values = [seed for values in second["partitions"].values() for seed in values]
        self.assertEqual(len(first_values), len(set(first_values)))
        self.assertTrue(set(first_values).isdisjoint(second_values))
        self.assertEqual(len(first["partitions"]["confirm_easy"]), 100)
        self.assertEqual(len(first["partitions"]["confirm_hard"]), 100)

    def test_method_freeze_gate_rejects_nonpositive_preservation(self) -> None:
        gate = self.protocol["method_development_gate"]
        summary = {
            "task": "place_container_plate",
            "stage": "brace_v2_intervention_pilot",
            "status": "passed",
            "complete": True,
            "preservation_go_no_go": {
                "passed": True,
                "paired_seed_count": 5,
                "effect_point_estimate": 0.0,
                "directional_wins": 5,
            },
        }
        errors = validate_method_pilot_summary(summary, gate)
        self.assertTrue(any("point estimate" in error for error in errors))

    def test_local_dp_reproduction_is_a_confirmatory_gate(self) -> None:
        protocol = json.loads(json.dumps(self.protocol))
        protocol["inference"]["bootstrap_replicates"] = 20
        task_manifest = {
            "heldout_tasks": ["task_a"],
            "tasks": {"task_a": {"leaderboard_dp_easy": 0.0}},
        }
        all_success = eval_payload({
            "preservation": [True, True],
            "confirm_easy": [True, True],
            "confirm_hard": [True, True],
        })
        matrix = {
            "task_a": {
                "base": {0: all_success},
                **{
                    arm: {seed: all_success for seed in protocol["training"]["training_seeds"]}
                    for arm in ("u0", "n1", "b1", "b2", "b3")
                },
            }
        }
        summary = aggregate_multitask(
            protocol,
            task_manifest,
            matrix,
            provenance={"complete": True},
        )
        self.assertFalse(summary["gates"]["local_dp_reproduction"])
        self.assertIn("local_dp_reproduction", summary["failed_gates"])

    def test_factorial_effect_directions(self) -> None:
        payloads = {arm: {} for arm in ("u0", "n1", "b1", "b2", "b3")}
        for seed in (1, 2):
            payloads["u0"][seed] = eval_payload({"preservation": [True] * 4, "confirm_easy": [True] * 4})
            payloads["n1"][seed] = eval_payload({"preservation": [False] * 4, "confirm_easy": [False] * 4})
            payloads["b1"][seed] = eval_payload({"preservation": [False] * 4, "confirm_easy": [True] * 4})
            payloads["b2"][seed] = eval_payload({"preservation": [True] * 4, "confirm_easy": [False] * 4})
            payloads["b3"][seed] = eval_payload({"preservation": [True] * 4, "confirm_easy": [True] * 4})
        effects = effect_maps_for_task(payloads, [1, 2])
        for seed in (1, 2):
            self.assertEqual(set(effects["anchor_main_preservation"][seed].values()), {1.0})
            self.assertEqual(set(effects["credit_main_adaptation"][seed].values()), {1.0})
            self.assertEqual(set(effects["brace_preservation"][seed].values()), {1.0})
            self.assertEqual(set(effects["brace_adaptation"][seed].values()), {0.0})

    def test_hierarchical_bootstrap_preserves_task_weighting(self) -> None:
        effects = {
            "task_a": {1: {1: 1.0, 2: 1.0}},
            "task_b": {1: {1: -1.0, 2: -1.0}},
        }
        summary = hierarchical_task_bootstrap(effects, replicates=200, seed=7, confidence_level=0.95)
        self.assertAlmostEqual(summary["point"], 0.0)

    def test_artifact_index_requires_u0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "eval.json"
            artifact.write_text(json.dumps(eval_payload({"preservation": [True]})), encoding="utf-8")
            entries = []
            for arm in ("base", "n1", "b1", "b2", "b3"):
                seeds = (0,) if arm == "base" else (1,)
                for seed in seeds:
                    entries.append({
                        "task": "task_a", "arm": arm, "training_seed": seed,
                        "path": str(artifact), "sha256": file_sha256(artifact),
                    })
            index = root / "index.json"
            index.write_text(json.dumps({"artifacts": entries}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "u0"):
                load_artifact_index(index, tasks=["task_a"], training_seeds=[1])

    def test_eval_payload_rejects_short_confirmatory_split(self) -> None:
        payload = eval_payload({"confirm_easy": [True] * 99, "confirm_hard": [True] * 100})
        payload["rows"].extend(
            {"split": "preservation", "env_seed": env_seed, "repeat": repeat, "success": True}
            for env_seed in range(100, 200)
            for repeat in range(3)
        )
        with self.assertRaisesRegex(ValueError, "confirm_easy"):
            validate_eval_payload(payload, self.protocol, label="short")

    def test_eval_payload_rejects_missing_success(self) -> None:
        payload = {"rows": [{"split": "preservation", "env_seed": 1, "repeat": 0}]}
        with self.assertRaisesRegex(ValueError, "success"):
            rows_by_key(payload, "preservation")

    def test_eval_payload_requires_frozen_partition_membership(self) -> None:
        payload = eval_payload({
            "preservation": [True] * 100,
            "confirm_easy": [True] * 100,
            "confirm_hard": [True] * 100,
        })
        payload["rows"] = [row for row in payload["rows"] if row["split"] != "preservation"]
        payload["rows"].extend(
            {"split": "preservation", "env_seed": env_seed, "repeat": repeat, "success": True}
            for env_seed in range(100, 200)
            for repeat in range(3)
        )
        manifest = {
            "partitions": {
                "census_candidate": list(range(200, 400)),
                "confirm_easy": list(range(100, 200)),
                "confirm_hard": list(range(100, 200)),
            }
        }
        with self.assertRaisesRegex(ValueError, "outside partition census_candidate"):
            validate_eval_payload(payload, self.protocol, label="leak", seed_manifest=manifest)

    def test_scheduler_rejects_wrong_simulator_packing(self) -> None:
        validation = validate_multitask_protocol(self.protocol_path)
        manifest = {
            "protocol_sha256": validation["protocol_sha256"],
            "task_manifest_sha256": validation["task_manifest_sha256"],
            "method_freeze_sha256": validation["method_freeze_sha256"],
            "jobs": [{
                "id": "eval:task", "kind": "simulator", "task": validation["heldout_tasks"][0],
                "command": ["true"], "dependencies": [], "artifact": "out.json", "workers_per_gpu": 1,
            }],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            result = validate_job_manifest(path, validation, validation["heldout_tasks"])
        self.assertFalse(result["passed"])
        self.assertTrue(any("workers_per_gpu=3" in error for error in result["errors"]))

    def test_scheduler_rejects_dependency_cycle(self) -> None:
        validation = validate_multitask_protocol(self.protocol_path)
        task = validation["heldout_tasks"][0]
        manifest = {
            "protocol_sha256": validation["protocol_sha256"],
            "task_manifest_sha256": validation["task_manifest_sha256"],
            "method_freeze_sha256": validation["method_freeze_sha256"],
            "jobs": [
                {"id": "a", "kind": "cpu", "task": task, "command": ["true"], "dependencies": ["b"], "artifact": "a.json"},
                {"id": "b", "kind": "cpu", "task": task, "command": ["true"], "dependencies": ["a"], "artifact": "b.json"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            result = validate_job_manifest(path, validation, validation["heldout_tasks"])
        self.assertFalse(result["passed"])
        self.assertIn("job dependency graph contains a cycle", result["errors"])

    def test_scheduler_finishes_independent_work_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            state = {
                "status": "running",
                "jobs": {
                    "failed": {"id": "failed", "status": "failed", "dependencies": []},
                    "independent": {"id": "independent", "status": "pending", "dependencies": []},
                },
            }
            scheduler = Scheduler(SimpleNamespace(gpus=[0], max_retries=0), state_path, state)

            def complete_launch(_gpu: int, job: dict) -> None:
                job["status"] = "completed"

            scheduler.launch = complete_launch  # type: ignore[method-assign]
            with mock.patch("experiments.brace.multitask_scheduler.signal.signal"), mock.patch(
                "experiments.brace.multitask_scheduler.time.sleep", return_value=None
            ):
                self.assertEqual(scheduler.run(), 1)
            self.assertEqual(state["jobs"]["independent"]["status"], "completed")
            self.assertEqual(state["status"], "failed")


if __name__ == "__main__":
    unittest.main()
