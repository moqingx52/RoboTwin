import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from experiments.brace.analyze_actor_failures import classify_failure, main, resolve_failures_path
from experiments.brace.collect_branches import (
    _select_control_failure_rollout_ids,
    bootstrap_lcb,
    evaluate_harness_sanity,
    select_branch_points,
    summarize_branch_rows,
)
from experiments.brace.control_trace import (
    build_branch_context,
    policy_chunk_index_at_physics_step,
    take_action_cnt_before_chunk,
)
from experiments.brace.replay_audit import Candidate


def _make_control_step(physics_step: int, policy_chunk_index: int, joint_value: float = 0.0) -> dict:
    robot_state = {
        "joints": np.asarray([joint_value], dtype=np.float64),
        "left_qpos": np.zeros(7, dtype=np.float64),
        "left_qvel": np.zeros(7, dtype=np.float64),
        "right_qpos": np.zeros(7, dtype=np.float64),
        "right_qvel": np.zeros(7, dtype=np.float64),
        "left_endpose": np.zeros(7, dtype=np.float64),
        "right_endpose": np.zeros(7, dtype=np.float64),
        "dynamic_actors": {},
    }
    return {
        "physics_step": int(physics_step),
        "policy_chunk_index": int(policy_chunk_index),
        "left_arm_pos": np.zeros(7, dtype=np.float64),
        "left_arm_vel": np.zeros(7, dtype=np.float64),
        "right_arm_pos": np.zeros(7, dtype=np.float64),
        "right_arm_vel": np.zeros(7, dtype=np.float64),
        "left_gripper": 0.0,
        "right_gripper": 0.0,
        "robot_state": robot_state,
    }


def _make_trace(*, chunk_sizes: tuple[int, ...]) -> dict:
    control_steps = []
    physics_step = 0
    for chunk_index, chunk_size in enumerate(chunk_sizes):
        for _ in range(chunk_size):
            control_steps.append(_make_control_step(physics_step, chunk_index, float(physics_step)))
            physics_step += 1

    policy_chunks = []
    for chunk_index, chunk_size in enumerate(chunk_sizes):
        start_physics_step = next(
            step["physics_step"] for step in control_steps if step["policy_chunk_index"] == chunk_index
        )
        policy_chunks.append(
            {
                "chunk_index": chunk_index,
                "policy_step": chunk_index,
                "physics_step": int(start_physics_step),
                "action": np.ones((chunk_size, 4), dtype=np.float64),
            }
        )

    snapshot_indices = [1, chunk_sizes[0] + 1, chunk_sizes[0] + chunk_sizes[1] + 1]
    branch_snapshots = []
    for snapshot_id, buffer_index in enumerate(snapshot_indices):
        step = control_steps[buffer_index]
        branch_snapshots.append(
            {
                "snapshot_id": snapshot_id,
                "physics_step": int(step["physics_step"]),
                "control_trace_offset": int(step["physics_step"]),
                "robot_state": step["robot_state"],
                "observation_joint_vector": np.asarray(step["robot_state"]["joints"], dtype=np.float64),
            }
        )

    return {
        "meta": {},
        "policy_chunks": policy_chunks,
        "control_steps": control_steps,
        "branch_snapshots": branch_snapshots,
    }


class BranchContextTest(unittest.TestCase):
    def test_policy_chunk_index_from_control_step(self):
        trace = _make_trace(chunk_sizes=(4, 4, 4, 4))
        self.assertEqual(policy_chunk_index_at_physics_step(trace, 0), 0)
        self.assertEqual(policy_chunk_index_at_physics_step(trace, 4), 1)
        with self.assertRaises(ValueError):
            policy_chunk_index_at_physics_step(trace, 999)

    def test_chunk_boundary_advance_from_mid_chunk_snapshot(self):
        trace = _make_trace(chunk_sizes=(4, 4, 4, 4))
        snapshot = trace["branch_snapshots"][0]
        context = build_branch_context(trace, snapshot)
        self.assertEqual(context.snapshot_physics_step, int(snapshot["physics_step"]))
        self.assertEqual(context.branch_chunk_index, 1)
        self.assertEqual(len(context.replay_steps), 2)
        self.assertEqual(context.boundary_physics_step, 4)

    def test_take_action_cnt_at_boundary(self):
        trace = _make_trace(chunk_sizes=(3, 4, 2, 3))
        self.assertEqual(take_action_cnt_before_chunk(trace, 0), 0)
        self.assertEqual(take_action_cnt_before_chunk(trace, 1), 3)
        self.assertEqual(take_action_cnt_before_chunk(trace, 2), 7)

    def test_runtime_state_matches_boundary(self):
        trace = _make_trace(chunk_sizes=(4, 4, 4, 4))
        context = build_branch_context(trace, trace["branch_snapshots"][1])
        self.assertEqual(context.runtime_state["physics_step"], context.boundary_physics_step)
        self.assertEqual(context.runtime_state["policy_chunk_index"], context.branch_chunk_index - 1)
        self.assertFalse(context.runtime_state["eval_success"])


class CollectBranchesTest(unittest.TestCase):
    def test_bootstrap_lcb_is_conservative(self):
        rng = __import__("random").Random(0)
        lcb = bootstrap_lcb([True, False, True], alpha=0.1, samples=200, rng=rng)
        self.assertLessEqual(lcb, 0.67)

    def test_summarize_branch_rows_detects_lift(self):
        rows = [
            {
                "env_seed": 1,
                "snapshot_id": 0,
                "physics_step": 10,
                "snapshot_physics_step": 8,
                "branch_chunk_index": 2,
                "point_type": "first_persistent_divergence",
                "branch_role": "candidate",
                "success": True,
            },
            {
                "env_seed": 1,
                "snapshot_id": 0,
                "physics_step": 10,
                "snapshot_physics_step": 8,
                "branch_chunk_index": 2,
                "point_type": "first_persistent_divergence",
                "branch_role": "control",
                "success": False,
            },
        ]
        summary = summarize_branch_rows(rows, alpha=0.1, delta=0.15)
        self.assertEqual(summary["matched_success_lift"], 1.0)
        self.assertTrue(summary["passed"])

    def test_select_branch_points_use_snapshots_only(self):
        success_trace = _make_trace(chunk_sizes=(4, 4, 4, 4))
        failure_trace = _make_trace(chunk_sizes=(4, 4, 4, 4))
        for step in failure_trace["control_steps"]:
            step["robot_state"]["joints"] = np.asarray([float(step["physics_step"]) + 0.5], dtype=np.float64)

        success = Candidate("task", 1, 0, True, Path("success.hdf5"))
        failure = Candidate("task", 1, 1, False, Path("failure.hdf5"))
        rng = __import__("random").Random(0)
        with mock.patch("experiments.brace.collect_branches.load_brace_trace") as loader:
            loader.side_effect = [success_trace, failure_trace]
            points = select_branch_points(
                "task",
                1,
                success,
                failure,
                max_points=3,
                rng=rng,
                success_trace=success_trace,
                failure_trace=failure_trace,
            )

        self.assertEqual(len(points), 3)
        snapshot_ids = {point.snapshot_id for point in points}
        self.assertEqual(len(snapshot_ids), 3)
        for point in points:
            self.assertNotEqual(point.physics_step, point.snapshot_physics_step)
            self.assertGreater(point.branch_chunk_index, 0)

    def test_control_same_chunk_different_rollouts(self):
        failures = [
            Candidate("task", 1, 1, False, Path("f1.hdf5")),
            Candidate("task", 1, 2, False, Path("f2.hdf5")),
            Candidate("task", 1, 3, False, Path("f3.hdf5")),
        ]
        trace = _make_trace(chunk_sizes=(4, 4, 4, 4))
        with mock.patch("experiments.brace.collect_branches.load_brace_trace", return_value=trace):
            rollout_ids = _select_control_failure_rollout_ids(
                failures,
                branch_chunk_index=1,
                control_k=3,
                rng=__import__("random").Random(0),
            )
        self.assertEqual(len(rollout_ids), 3)
        self.assertEqual(len(set(rollout_ids)), 3)

    def test_harness_sanity_marks_invalid(self):
        rows = [
            {
                "branch_role": "candidate",
                "success": False,
            }
        ]
        valid, reason = evaluate_harness_sanity(rows, min_rate=0.05)
        self.assertFalse(valid)
        self.assertIn("below_0.05", reason or "")


class AnalyzeActorFailuresTest(unittest.TestCase):
    def test_classify_garbage_rotation(self):
        row = {
            "actor_metric_passed": {"garbage_1": {"translation": True, "rotation": False}},
            "errors": {"actor_errors": {"garbage_1": {"rotation_error": 0.1}}},
        }
        self.assertEqual(classify_failure(row), "garbage_rotation")

    def test_cli_writes_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            audit_dir = Path(directory) / "audit"
            audit_dir.mkdir()
            failures = audit_dir / "failures.jsonl"
            failures.write_text(
                json.dumps(
                    {
                        "check_type": "control_trace_replay",
                        "passed": False,
                        "worst_actor": "garbage_0",
                        "worst_metric": "rotation",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = Path(directory) / "summary.json"
            with mock.patch(
                "sys.argv",
                ["analyze", "--audit-dir", str(audit_dir), "--output", str(output)],
            ):
                self.assertEqual(main(), 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(payload["garbage_rotation_dominates"])

    def test_resolve_failures_json_to_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            audit_dir = Path(directory)
            (audit_dir / "failures.jsonl").write_text(
                json.dumps({"check_type": "control_trace_replay", "passed": False}) + "\n",
                encoding="utf-8",
            )
            path = resolve_failures_path(audit_dir / "failures.json", None)
            self.assertEqual(path.name, "failures.jsonl")


if __name__ == "__main__":
    unittest.main()
