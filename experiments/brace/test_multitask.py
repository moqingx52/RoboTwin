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
from experiments.brace.aggregate_seed_feasibility_batch import build_batch_summary
from experiments.brace.diagnose_line_b_collect_parity import (
    nearby_control_indices,
    parse_failure_specs,
    summarize as summarize_collect_parity,
)
from experiments.brace.diagnose_handover_mic import resolve_seed_cases as resolve_handover_seed_cases
from experiments.brace.multitask_protocol import (
    SUPPLEMENT_PARTITION_NAME,
    build_seed_manifest,
    file_sha256,
    validate_method_pilot_summary,
    validate_multitask_amendment,
    validate_multitask_protocol,
)
from experiments.brace.multitask_scheduler import Scheduler, validate_job_manifest
from experiments.brace.prepare_multitask_demo_seeds import check_manifest_frozen
from experiments.brace.premotion_retry import (
    materialize_saved_seed_trajectory,
    resolve_max_attempts,
    validate_retry_amendment,
)
from experiments.brace.scan_seed_feasibility import merge_probe_results, shard_slice
from experiments.brace.seed_feasibility import (
    PROBE_SUCCESS_RULE_MAJORITY,
    TASK_STATUS_EXPERT_EXCEPTION,
    TASK_STATUS_INSUFFICIENT,
    TASK_STATUS_PASSED,
    apply_provisional_expert_demo_to_manifest,
    build_feasibility_evidence,
    derive_task_status,
    expert_demo_source_pool,
    probe_repeat_outcome,
    select_expert_demo_seeds,
    validate_feasibility_evidence,
    validate_seed_manifest_layout,
)


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

    def test_collect_parity_failure_specs_and_nearby_controls(self) -> None:
        parsed = parse_failure_specs(["task_a=12", "task_a=15", "task_b=21", "task_a=12"])
        self.assertEqual(parsed, {"task_a": [12, 15], "task_b": [21]})
        self.assertEqual(nearby_control_indices(list(range(10)), [4, 6], count=4), [3, 5, 7, 2])

    def test_collect_parity_summary_splits_failure_modes(self) -> None:
        rows = [
            {"task": "task_a", "role": "reported_failure", "passed": True, "error_type": None},
            {"task": "task_a", "role": "reported_failure", "passed": False, "error_type": "expert_plan_failed"},
            {"task": "task_a", "role": "nearby_control", "passed": False, "error_type": "expert_check_failed"},
        ]
        summary = summarize_collect_parity(rows)
        self.assertEqual(summary["task_a"]["reported_failure"]["attempts"], 2)
        self.assertEqual(summary["task_a"]["reported_failure"]["passed"], 1)
        self.assertEqual(summary["task_a"]["reported_failure"]["plan_failed"], 1)
        self.assertEqual(summary["task_a"]["nearby_control"]["check_failed"], 1)

    def test_handover_diagnostic_uses_real_cohort_indices(self) -> None:
        manifest = {"cohorts": {"expert_demo": [140000, 140004, 140018]}}
        self.assertEqual(
            resolve_handover_seed_cases(manifest, [140004, 140018]),
            [{"seed": 140004, "episode_idx": 1}, {"seed": 140018, "episode_idx": 2}],
        )
        with self.assertRaisesRegex(ValueError, "not in"):
            resolve_handover_seed_cases(manifest, [140005])

    def test_bounded_premotion_retry_logs_fail_then_success(self) -> None:
        class FakeEnv:
            def __init__(self) -> None:
                self.attempt = 0
                self.plan_success = False

            def setup_demo(self, **_kwargs) -> None:
                self.attempt += 1

            def play_once(self) -> None:
                self.plan_success = self.attempt >= 2

            def check_success(self) -> bool:
                return True

            def save_traj_data(self, episode_idx: int) -> None:
                path = Path(self.save_path) / "_traj_data"
                path.mkdir(parents=True, exist_ok=True)
                (path / f"episode{episode_idx}.pkl").write_bytes(b"trajectory")

            def close_env(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as tmp:
            env = FakeEnv()
            env.save_path = tmp
            args = {"save_path": tmp, "task_name": "task_a", "task_config": "demo_clean", "render_freq": 0}
            result = materialize_saved_seed_trajectory(
                env,
                args,
                seed=10,
                episode_idx=3,
                max_attempts=2,
                amendment_path="amendment.json",
            )
            rows = [json.loads(line) for line in (Path(tmp) / "premotion_attempts.jsonl").read_text().splitlines()]
            self.assertEqual(result["attempt"], 2)
            self.assertEqual([row["passed"] for row in rows], [False, True])
            self.assertEqual(rows[0]["error_type"], "expert_plan_failed")

    def test_retry_above_one_requires_matching_frozen_amendment(self) -> None:
        payload = {
            "status": "frozen",
            "operational_collection_retry": {
                "eligible_tasks": ["task_a"],
                "max_attempts_per_episode": 5,
                "seed_substitution": False,
                "episode_index_rule": "frozen_cohort_index",
                "acceptance_rule": "plan_success_and_check_success_and_trajectory_saved",
                "log_all_attempts": True,
            },
        }
        self.assertEqual(validate_retry_amendment(payload, max_attempts=5), [])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "amendment.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            env = {
                "BRACE_PREMOTION_MAX_ATTEMPTS": "5",
                "BRACE_PREMOTION_RETRY_AMENDMENT": str(path),
            }
            with mock.patch("experiments.brace.premotion_retry.validate_amendment_bindings", return_value=[]):
                self.assertEqual(resolve_max_attempts(env, task_name="task_a")[0], 5)
                with self.assertRaisesRegex(RuntimeError, "not eligible"):
                    resolve_max_attempts(env, task_name="task_b")

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

    def test_expert_demo_selection_keeps_rollout_train_order(self) -> None:
        candidates = [10, 11, 12, 13, 14]
        results = [
            {"seed": 10, "passed": False},
            {"seed": 11, "passed": True},
            {"seed": 12, "passed": False},
            {"seed": 13, "passed": True},
            {"seed": 14, "passed": True},
        ]
        selected = select_expert_demo_seeds(candidates, results, required_count=2)
        self.assertEqual(selected, [11, 13])

    def test_expert_demo_evidence_applies_subset_without_reordering_rollout_train(self) -> None:
        manifest = {
            "partitions": {
                "rollout_train": [10, 11, 12, 13],
                "anchor_candidate": [100],
            },
            "cohorts": {"expert_demo": []},
            "feasibility": {"passed": False},
        }
        evidence = build_feasibility_evidence(
            task="task_a",
            candidate_seeds=[10, 11, 12, 13],
            probe_results=[
                {"seed": 10, "passed": False},
                {"seed": 11, "passed": True},
                {"seed": 12, "passed": False},
                {"seed": 13, "passed": True},
            ],
            required_count=2,
        )
        self.assertEqual(validate_feasibility_evidence(evidence), [])
        updated = apply_provisional_expert_demo_to_manifest(manifest, evidence)
        self.assertEqual(updated["partitions"]["rollout_train"], [10, 11, 12, 13])
        self.assertNotIn("expert_demo", updated["partitions"])
        self.assertEqual(updated["cohorts"]["expert_demo"], [11, 13])
        self.assertEqual(updated["status"], "candidate_unvalidated")
        self.assertTrue(updated["expert_demo_selection"]["provisional"])
        self.assertEqual(
            updated["expert_demo_selection"]["rule"],
            "first_n_solvable_in_manifest_order",
        )

    def test_evidence_schema_v2_records_provenance_and_determinism(self) -> None:
        evidence = build_feasibility_evidence(
            task="task_a",
            candidate_seeds=[10, 11, 12, 13],
            probe_results=[
                {"seed": 10, "passed": True, "probe_repeats": 3, "probe_passed_count": 2},
                {"seed": 11, "passed": True, "probe_repeats": 3, "probe_passed_count": 3},
                {"seed": 12, "passed": False, "probe_repeats": 3, "probe_passed_count": 1},
                {"seed": 13, "passed": True, "probe_repeats": 3, "probe_passed_count": 2},
            ],
            required_count=2,
            gpus=["3"],
            shard_count=2,
            probe_repeats=3,
            probe_success_rule=PROBE_SUCCESS_RULE_MAJORITY,
        )
        self.assertEqual(evidence["schema_version"], 2)
        self.assertEqual(evidence["task_status"], TASK_STATUS_PASSED)
        self.assertEqual(evidence["gpus"], ["3"])
        self.assertEqual(evidence["shard_count"], 2)
        self.assertEqual(evidence["determinism"]["probe_repeats"], 3)
        provenance = evidence["provenance"]
        self.assertTrue(provenance["code"]["code_commit"])
        self.assertTrue(provenance["task_config_sha256"])
        self.assertEqual(validate_feasibility_evidence(evidence), [])

    def test_validator_rejects_passed_evidence_without_task_status(self) -> None:
        evidence = build_feasibility_evidence(
            task="task_a",
            candidate_seeds=[10, 11, 12, 13],
            probe_results=[
                {"seed": 10, "passed": False},
                {"seed": 11, "passed": True},
                {"seed": 12, "passed": False},
                {"seed": 13, "passed": True},
            ],
            required_count=2,
        )
        evidence.pop("task_status")
        errors = validate_feasibility_evidence(evidence)
        self.assertTrue(any("task_status" in error for error in errors))

    def test_validator_rejects_duplicate_result_seeds(self) -> None:
        evidence = build_feasibility_evidence(
            task="task_a",
            candidate_seeds=[10, 11],
            probe_results=[
                {"seed": 10, "passed": True},
                {"seed": 10, "passed": False},
            ],
            required_count=1,
        )
        errors = validate_feasibility_evidence(evidence)
        self.assertTrue(any("duplicate" in error for error in errors))

    def test_manifest_layout_rejects_partition_overlap_and_deprecated_expert_demo(self) -> None:
        errors = validate_seed_manifest_layout({
            "partitions": {
                "rollout_train": [10, 11, 12],
                "anchor_candidate": [12, 13],
            },
        })
        self.assertTrue(any("overlap" in error for error in errors))
        errors = validate_seed_manifest_layout({
            "partitions": {
                "rollout_train": [10, 11, 12],
                "expert_demo": [10, 11],
            },
        })
        self.assertTrue(any("deprecated" in error for error in errors))
        self.assertEqual(
            validate_seed_manifest_layout({
                "partitions": {"rollout_train": [10, 11], "anchor_candidate": [100]},
                "cohorts": {"expert_demo": [10]},
            }),
            [],
        )
        errors = validate_seed_manifest_layout({
            "partitions": {"rollout_train": [10, 11]},
            "cohorts": {"expert_demo": [10, 999]},
        })
        self.assertTrue(any("expert_demo_supplement" in error for error in errors))

    def test_seed_manifest_reserves_supplement_partition(self) -> None:
        protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        amendment = json.loads((BRACE_DIR / "multitask_amendment.v1.1.json").read_text(encoding="utf-8"))
        manifest = build_seed_manifest(protocol, self.tasks, "lift_pot", amendment=amendment)
        supplement = manifest["partitions"]["expert_demo_supplement"]
        self.assertEqual(len(supplement), 200)
        base = 150000
        self.assertEqual(supplement, list(range(base + 700, base + 900)))
        flat = [seed for values in manifest["partitions"].values() for seed in values]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(
            expert_demo_source_pool(manifest),
            manifest["partitions"]["rollout_train"] + manifest["partitions"]["expert_demo_supplement"],
        )

    def test_amendment_validates_and_binds_original_evidence(self) -> None:
        amendment_path = BRACE_DIR / "multitask_amendment.v1.1.json"
        protocol_path = BRACE_DIR / "multitask_protocol.v1.json"
        self.assertEqual(validate_multitask_amendment(amendment_path, protocol_path), [])
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        tampered = json.loads(json.dumps(amendment))
        tampered["supplement"]["partition_name"] = "other_pool"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "amendment.json"
            path.write_text(json.dumps(tampered), encoding="utf-8")
            Path(str(path) + ".sha256").write_text(
                f"{file_sha256(path)}  amendment.json\n", encoding="utf-8"
            )
            errors = validate_multitask_amendment(path, protocol_path)
        self.assertTrue(any("partition_name" in error for error in errors))
        tampered = json.loads(json.dumps(amendment))
        tampered["applies_to"]["protocol_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "amendment.json"
            path.write_text(json.dumps(tampered), encoding="utf-8")
            Path(str(path) + ".sha256").write_text(
                f"{file_sha256(path)}  amendment.json\n", encoding="utf-8"
            )
            errors = validate_multitask_amendment(path, protocol_path)
        self.assertTrue(any("protocol_sha256" in error for error in errors))
        tampered = json.loads(json.dumps(amendment))
        del tampered["original_evidence"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "amendment.json"
            path.write_text(json.dumps(tampered), encoding="utf-8")
            Path(str(path) + ".sha256").write_text(
                f"{file_sha256(path)}  amendment.json\n", encoding="utf-8"
            )
            errors = validate_multitask_amendment(path, protocol_path)
        self.assertTrue(any("original_evidence" in error for error in errors))

    def test_combined_supplement_evidence_validates_and_sets_manifest_flag(self) -> None:
        base = list(range(10, 15))
        base_results = [
            {"seed": seed, "passed": seed in (10, 11, 12, 13)}
            for seed in base
        ]
        original = build_feasibility_evidence(
            task="task_a",
            candidate_seeds=base,
            probe_results=base_results,
            required_count=5,
        )
        self.assertEqual(original["task_status"], TASK_STATUS_INSUFFICIENT)
        supplement = list(range(100, 106))
        supplement_results = [{"seed": seed, "passed": True} for seed in supplement]
        combined = build_feasibility_evidence(
            task="task_a",
            candidate_seeds=base + supplement,
            probe_results=base_results + supplement_results,
            required_count=5,
            supplement={
                "partition": "expert_demo_supplement",
                "count": len(supplement),
                "original_pool_solvable": 4,
                "original_pool_candidate_count": len(base),
                "original_pool_task_status": TASK_STATUS_INSUFFICIENT,
                "original_evidence_path": "runs/run/task_a_feasibility.json",
                "original_evidence_sha256": "a" * 64,
                "selection_rule": "original_pool_first_then_supplement_ascending",
                "supplement_solvable": len(supplement),
                "supplement_shard_count": 1,
                "used": True,
                "supplement_seeds_used": [100],
            },
        )
        self.assertEqual(combined["task_status"], TASK_STATUS_PASSED)
        self.assertEqual(combined["expert_demo_seeds"], [10, 11, 12, 13, 100])
        self.assertEqual(validate_feasibility_evidence(combined), [])
        manifest = {
            "partitions": {
                "rollout_train": base,
                "expert_demo_supplement": supplement,
            },
            "cohorts": {"expert_demo": []},
            "feasibility": {"passed": False},
        }
        updated = apply_provisional_expert_demo_to_manifest(manifest, combined)
        self.assertEqual(updated["cohorts"]["expert_demo"], [10, 11, 12, 13, 100])
        self.assertTrue(updated["expert_demo_selection"]["supplement_used"])

    def test_classify_probe_error_splits_plan_and_check(self) -> None:
        from experiments.brace.seed_feasibility import classify_probe_error

        plan_exc = RuntimeError("Saved seed 1 expert plan failed for episode 0")
        check_exc = RuntimeError("Saved seed 1 expert success check failed for episode 0")
        legacy_exc = RuntimeError("Saved seed 1 failed pre-motion validation for episode 0")
        self.assertEqual(classify_probe_error(plan_exc)[0], "expert_plan_failed")
        self.assertEqual(classify_probe_error(check_exc)[0], "expert_check_failed")
        self.assertEqual(classify_probe_error(legacy_exc)[0], "pre_motion_validation_failed")

    def test_probe_passed_strict_majority_never_degrades_to_any(self) -> None:
        from experiments.brace.seed_feasibility import probe_passed
        self.assertFalse(probe_passed(1, 2, "majority"))
        self.assertTrue(probe_passed(2, 2, "majority"))
        self.assertFalse(probe_passed(1, 3, "majority"))
        self.assertTrue(probe_passed(2, 3, "majority"))
        self.assertFalse(probe_passed(1, 2, "all"))
        self.assertTrue(probe_passed(2, 2, "all"))

    def test_validator_rejects_combine_determinism_drift(self) -> None:
        evidence = build_feasibility_evidence(
            task="task_a",
            candidate_seeds=[10, 11],
            probe_results=[
                {"seed": 10, "passed": True, "probe_repeats": 2, "probe_passed_count": 2},
                {"seed": 11, "passed": True, "probe_repeats": 2, "probe_passed_count": 2},
            ],
            required_count=2,
        )
        evidence["determinism"] = {"probe_repeats": 1, "probe_success_rule": "single"}
        errors = validate_feasibility_evidence(evidence)
        self.assertTrue(any("probe_repeats=2" in error and "determinism.probe_repeats=1" in error for error in errors))

    def test_validator_rejects_any_as_majority_passed_flag(self) -> None:
        evidence = build_feasibility_evidence(
            task="task_a",
            candidate_seeds=[10, 11],
            probe_results=[
                {"seed": 10, "passed": True, "probe_repeats": 2, "probe_passed_count": 1},
                {"seed": 11, "passed": True, "probe_repeats": 2, "probe_passed_count": 2},
            ],
            required_count=2,
        )
        evidence["determinism"] = {"probe_repeats": 2, "probe_success_rule": "majority"}
        errors = validate_feasibility_evidence(evidence)
        self.assertTrue(any("passed flag is inconsistent" in error for error in errors))

    def test_derive_corrected_recomputes_from_saved_probes(self) -> None:
        from experiments.brace.derive_corrected_feasibility import build_corrected_evidence
        candidates = [10, 11, 12, 13]
        rows = [
            {"seed": seed, "episode_idx": 0,
             "probe_repeats": 2,
             "probe_passed_count": passed,
             "probes": [
                 {"seed": seed, "passed": True if idx < passed else False,
                  "error_type": None, "error_message": None}
                 for idx in range(2)
             ]}
            for seed, passed in zip(candidates, (2, 1, 0, 2))
        ]
        source = build_feasibility_evidence(
            task="task_a",
            candidate_seeds=candidates,
            probe_results=[{**row, "passed": row["probe_passed_count"] >= 1} for row in rows],
            required_count=2,
        )
        source["determinism"] = {"probe_repeats": 2, "probe_success_rule": "majority"}
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "task_a_feasibility.json"
            src.write_text(json.dumps(source), encoding="utf-8")
            corrected = build_corrected_evidence(
                source, src, probe_repeats=2, probe_success_rule="all", required_count=2
            )
            source_sha = file_sha256(src)
        self.assertEqual(corrected["solvable_count_in_candidates"], 2)
        self.assertEqual(corrected["expert_demo_seeds"], [10, 13])
        self.assertEqual(corrected["determinism"], {"probe_repeats": 2, "probe_success_rule": "all"})
        self.assertEqual(validate_feasibility_evidence(corrected), [])
        self.assertEqual(
            corrected["provenance"]["source_evidence"]["sha256"],
            source_sha,
        )
        self.assertIn("derivation", corrected)

    def test_derive_corrected_refuses_rows_without_probe_data(self) -> None:
        from experiments.brace.derive_corrected_feasibility import build_corrected_evidence
        source = build_feasibility_evidence(
            task="task_a",
            candidate_seeds=[10, 11],
            probe_results=[
                {"seed": 10, "passed": True},
                {"seed": 11, "passed": False},
            ],
            required_count=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "task_a_feasibility.json"
            src.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot recompute offline"):
                build_corrected_evidence(
                    source, src, probe_repeats=2, probe_success_rule="all", required_count=2
                )

    def test_freeze_writes_frozen_manifest_with_valid_sidecar(self) -> None:
        from experiments.brace.freeze_multitask_manifest import freeze_manifest
        candidates = list(range(10, 14))
        evidence = build_feasibility_evidence(
            task="task_a",
            candidate_seeds=candidates,
            probe_results=[
                {"seed": 10, "passed": True, "probe_repeats": 2, "probe_passed_count": 2},
                {"seed": 11, "passed": True, "probe_repeats": 2, "probe_passed_count": 2},
                {"seed": 12, "passed": False, "probe_repeats": 2, "probe_passed_count": 0},
                {"seed": 13, "passed": False, "probe_repeats": 2, "probe_passed_count": 0},
            ],
            required_count=2,
        )
        evidence["determinism"] = {"probe_repeats": 2, "probe_success_rule": "all"}
        manifest = {
            "schema_version": 1,
            "task": "task_a",
            "status": "candidate_unvalidated",
            "cohorts": {"expert_demo": []},
            "partitions": {"rollout_train": candidates},
            "feasibility": {"passed": False},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ev_path = root / "task_a_feasibility_corrected.json"
            ev_path.write_text(json.dumps(evidence), encoding="utf-8")
            manifest_path = root / "task_a.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = freeze_manifest(
                "task_a",
                ev_path,
                manifest_path,
                required_repeats=2,
                required_rule="all",
                required_count=2,
            )
            frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(frozen["status"], "frozen")
            self.assertTrue(frozen["feasibility"]["passed"])
            self.assertEqual(frozen["cohorts"]["expert_demo"], [10, 11])
            sidecar = Path(str(manifest_path) + ".sha256")
            self.assertEqual(
                sidecar.read_text().split()[0],
                file_sha256(manifest_path),
            )
            self.assertEqual(result["evidence_sha256"], file_sha256(ev_path))
            tampered = json.loads(json.dumps(evidence))
            tampered["determinism"] = {"probe_repeats": 1, "probe_success_rule": "single"}
            bad_path = root / "bad.json"
            bad_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "determinism"):
                freeze_manifest(
                    "task_a", bad_path, manifest_path,
                    required_repeats=2, required_rule="all", required_count=2,
                )

    def test_batch_summary_prefers_supplemented_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task_a_feasibility.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
            evidence = build_feasibility_evidence(
                task="task_a",
                candidate_seeds=[10, 11],
                probe_results=[
                    {"seed": 10, "passed": True},
                    {"seed": 11, "passed": True},
                ],
                required_count=2,
            )
            (root / "task_a_supplemented_feasibility.json").write_text(json.dumps(evidence), encoding="utf-8")
            summary = build_batch_summary(root, ["task_a"])
            self.assertTrue(
                summary["tasks"]["task_a"]["evidence_path"].endswith("task_a_supplemented_feasibility.json")
            )

    def test_batch_summary_prefers_corrected_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task_a_supplemented_feasibility.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
            evidence = build_feasibility_evidence(
                task="task_a",
                candidate_seeds=[10, 11],
                probe_results=[
                    {"seed": 10, "passed": True},
                    {"seed": 11, "passed": True},
                ],
                required_count=2,
            )
            (root / "task_a_supplemented_feasibility_corrected.json").write_text(json.dumps(evidence), encoding="utf-8")
            summary = build_batch_summary(root, ["task_a"])
            self.assertTrue(
                summary["tasks"]["task_a"]["evidence_path"].endswith("task_a_supplemented_feasibility_corrected.json")
            )

    def test_probe_repeat_outcome_majority_rule(self) -> None:
        task_env = mock.Mock()
        task_env.setup_demo = mock.Mock()
        task_env.close_env = mock.Mock()
        task_env.play_once = mock.Mock()
        outcomes = iter([(True, None), (False, RuntimeError("boom")), (False, RuntimeError("boom"))])

        def fake_play_once() -> None:
            task_env.plan_success, _ = next(outcomes)

        task_env.play_once.side_effect = fake_play_once
        task_env.check_success = mock.Mock(return_value=True)
        row = probe_repeat_outcome(
            task_env,
            {"render_freq": 0},
            7,
            episode_idx=0,
            probe_repeats=3,
            success_rule=PROBE_SUCCESS_RULE_MAJORITY,
        )
        self.assertFalse(row["passed"])
        self.assertEqual(row["probe_passed_count"], 1)
        self.assertEqual(row["probe_repeats"], 3)
        self.assertIn("probes", row)

    def _write_shard(self, directory: Path, shard_id: int, shard_count: int, candidates: list[int], **overrides) -> Path:
        start, stop = shard_slice(len(candidates), shard_id, shard_count)
        payload = {
            "task": "task_a",
            "verify_label": None,
            "shard_id": shard_id,
            "num_shards": shard_count,
            "candidate_start": start,
            "candidate_end": stop,
            "gpu": "0",
            "probe_repeats": 1,
            "probe_success_rule": "single",
            "results": [
                {"seed": candidates[index], "episode_idx": 0, "passed": index % 2 == 0,
                 "error_type": None, "error_message": None}
                for index in range(start, stop)
            ],
        }
        payload.update(overrides)
        path = directory / f"shard_{shard_id:02d}_of_{shard_count:02d}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_shard_merge_requires_exact_coverage(self) -> None:
        candidates = list(range(10, 14))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = [
                self._write_shard(root, 0, 2, candidates),
                self._write_shard(root, 1, 2, candidates),
            ]
            merged = merge_probe_results(shards, candidate_seeds=candidates, task="task_a")
            self.assertEqual([int(row["seed"]) for row in merged], candidates)
            shards.append(self._write_shard(root, 0, 2, candidates))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                merge_probe_results(shards, candidate_seeds=candidates, task="task_a")
            self._write_shard(root, 2, 3, candidates)
            with self.assertRaisesRegex(ValueError, "disagree"):
                merge_probe_results(
                    [root / f for f in sorted(p.name for p in root.glob("shard_*.json"))],
                    candidate_seeds=candidates,
                    task="task_a",
                )

    def test_shard_merge_rejects_stale_shard_and_seed_mismatch(self) -> None:
        candidates = list(range(10, 14))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stale = self._write_shard(root, 0, 1, candidates)
            fresh = self._write_shard(root, 0, 1, candidates, num_shards=2, candidate_start=0, candidate_end=3)
            with self.assertRaisesRegex(ValueError, "expected"):
                merge_probe_results([stale, fresh], candidate_seeds=candidates, task="task_a")
            wrong_rows = [{"seed": 99, "episode_idx": 0, "passed": True, "error_type": None, "error_message": None}]
            wrong_rows.extend(
                {"seed": seed, "episode_idx": 0, "passed": True, "error_type": None, "error_message": None}
                for seed in candidates[1:]
            )
            wrong = self._write_shard(root, 0, 1, candidates, results=wrong_rows)
            with self.assertRaisesRegex(ValueError, "seed order mismatch"):
                merge_probe_results([wrong], candidate_seeds=candidates, task="task_a")

    def test_batch_summary_prefers_derived_v2_evidence(self) -> None:
        candidates = [10, 11]
        evidence = build_feasibility_evidence(
            task="task_a",
            candidate_seeds=candidates,
            probe_results=[
                {"seed": 10, "passed": True},
                {"seed": 11, "passed": True},
            ],
            required_count=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task_a_feasibility.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
            (root / "task_a_feasibility_v2.json").write_text(json.dumps(evidence), encoding="utf-8")
            summary = build_batch_summary(root, ["task_a"])
            self.assertTrue(summary["tasks"]["task_a"]["passed"])
            self.assertTrue(
                summary["tasks"]["task_a"]["evidence_path"].endswith("task_a_feasibility_v2.json")
            )
            self.assertTrue(summary["tasks"]["task_a"]["evidence_sha256"])

    def test_batch_summary_rejects_evidence_missing_task_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task_a_feasibility.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
            summary = build_batch_summary(root, ["task_a"])
            self.assertFalse(summary["tasks"]["task_a"]["passed"])
            self.assertTrue(
                any("task_status" in error for error in summary["tasks"]["task_a"]["invalid_evidence"])
            )

    def _write_frozen_manifest(self, directory: Path) -> Path:
        manifest = {
            "schema_version": 1,
            "task": "task_a",
            "status": "frozen",
            "cohorts": {"expert_demo": [10, 11]},
            "partitions": {"rollout_train": [10, 11, 12, 13]},
            "feasibility": {
                "criterion": "expert_script_simulator_solvability_only",
                "learning_policy_performance_consulted": False,
                "evidence_path": str(directory / "task_a_feasibility.json"),
                "evidence_sha256": None,
                "passed": True,
            },
        }
        evidence = build_feasibility_evidence(
            task="task_a",
            candidate_seeds=[10, 11, 12, 13],
            probe_results=[
                {"seed": 10, "passed": True},
                {"seed": 11, "passed": True},
                {"seed": 12, "passed": False},
                {"seed": 13, "passed": False},
            ],
            required_count=2,
        )
        evidence_path = directory / "task_a_feasibility.json"
        evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        manifest["feasibility"]["evidence_sha256"] = file_sha256(evidence_path)
        manifest_path = directory / "task_a.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        sidecar = directory / "task_a.json.sha256"
        sidecar.write_text(f"{file_sha256(manifest_path)}  task_a.json\n", encoding="utf-8")
        return manifest_path

    def test_frozen_gate_accepts_frozen_manifest_with_valid_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_frozen_manifest(Path(tmp))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            check_manifest_frozen(manifest, manifest_path)

    def test_frozen_gate_rejects_unfrozen_or_invalid_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_frozen_manifest(Path(tmp))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "candidate_unvalidated"
            with self.assertRaises(SystemExit):
                check_manifest_frozen(manifest, manifest_path)
            manifest["status"] = "frozen"
            manifest["feasibility"]["passed"] = False
            with self.assertRaises(SystemExit):
                check_manifest_frozen(manifest, manifest_path)
            manifest["feasibility"]["passed"] = True
            manifest["feasibility"]["evidence_sha256"] = "0" * 64
            with self.assertRaisesRegex(SystemExit, "SHA256"):
                check_manifest_frozen(manifest, manifest_path)

    def test_derive_task_status_marks_expert_assertions_separately(self) -> None:
        results = [
            {"seed": 1, "passed": False, "error_type": "expert_assertion_failed"},
            {"seed": 2, "passed": False, "error_type": "pre_motion_validation_failed"},
        ]
        self.assertEqual(
            derive_task_status(results, candidate_count=2, required_count=50),
            TASK_STATUS_EXPERT_EXCEPTION,
        )
        results = [
            {"seed": 1, "passed": False, "error_type": "pre_motion_validation_failed"},
            {"seed": 2, "passed": False, "error_type": "simulator_unstable"},
        ]
        self.assertEqual(
            derive_task_status(results, candidate_count=2, required_count=50),
            TASK_STATUS_INSUFFICIENT,
        )
        results = [{"seed": i, "passed": True} for i in range(50)]
        self.assertEqual(
            derive_task_status(results, candidate_count=50, required_count=50),
            TASK_STATUS_PASSED,
        )


if __name__ == "__main__":
    unittest.main()
