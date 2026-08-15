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
    paired_cluster_lcb,
    select_branch_points,
    summarize_branch_rows,
)
from experiments.brace.control_trace import (
    build_branch_context,
    policy_chunk_index_at_physics_step,
    take_action_cnt_before_chunk,
)
from experiments.brace.replay_audit import Candidate
from experiments.brace.export_verified_chunks import branch_confirm_completeness, exportable_point_types


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
    def test_branch_confirm_gate_is_hard_and_random_controls_are_not_exportable(self):
        protocol = {
            "branch_confirm_gate": {
                "min_seeds": 10,
                "min_points": 30,
                "max_seed_accepted_share": 0.2,
            }
        }
        pilot = [
            {"env_seed": seed, "accepted": True}
            for seed in range(3)
            for _ in range(3)
        ]
        result = branch_confirm_completeness(pilot, protocol)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["min_seeds"])
        self.assertFalse(result["checks"]["min_points"])
        self.assertNotIn("random_negative_control", exportable_point_types(protocol))

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

    def test_paired_cluster_lcb_does_not_degenerate_for_three_of_three(self):
        rows = []
        for continuation_seed in range(3):
            rows.extend(
                [
                    {
                        "branch_role": "candidate",
                        "continuation_seed": continuation_seed,
                        "success": True,
                    },
                    {
                        "branch_role": "control",
                        "continuation_seed": continuation_seed,
                        "success": False,
                    },
                ]
            )
        rng = __import__("random").Random(0)
        lcb, paired = paired_cluster_lcb(rows, alpha=0.1, samples=500, rng=rng)
        self.assertEqual(paired, [1.0, 1.0, 1.0])
        self.assertLess(lcb, 1.0)
        self.assertLessEqual(lcb, 0.15)

    def test_select_branch_points_causal_before_divergence_growth(self):
        # Snapshots sit at physics steps 1, 5, 9 (chunks 0, 1, 2). The failure
        # tracks the success closely until step 9, where it jumps: consensus
        # divergences at the snapshots are [0.1, 0.1, 5.0], so the largest
        # growth is between snapshot 1 and snapshot 2 (k* = 1) and the causal
        # point must branch one boundary earlier, at snapshot 0 (k* - 1).
        success_trace = _make_trace(chunk_sizes=(4, 4, 4, 4))
        failure_trace = _make_trace(chunk_sizes=(4, 4, 4, 4))
        for step in failure_trace["control_steps"]:
            offset = 0.1 if step["physics_step"] < 9 else 5.0
            step["robot_state"]["joints"] = np.asarray(
                [float(step["physics_step"]) + offset], dtype=np.float64
            )

        success = Candidate("task", 1, 0, True, Path("success.hdf5"))
        failure = Candidate("task", 1, 1, False, Path("failure.hdf5"))
        rng = __import__("random").Random(0)
        points = select_branch_points(
            "task",
            1,
            success,
            [failure],
            max_points=2,
            rng=rng,
            success_trace=success_trace,
            failure_traces=[failure_trace],
        )

        self.assertEqual(len(points), 2)
        self.assertEqual(
            {point.point_type for point in points},
            {"pre_divergence_causal", "matched_random_control"},
        )
        causal = next(p for p in points if p.point_type == "pre_divergence_causal")
        control = next(p for p in points if p.point_type == "matched_random_control")
        self.assertEqual(causal.snapshot_id, 0)
        self.assertEqual(causal.branch_chunk_index, 1)
        # With only 3 snapshots the growth window {k*-1, k*, k*+1} covers the
        # whole pool, so the control falls back to any non-causal snapshot.
        self.assertIn(control.snapshot_id, {1, 2})
        for point in points:
            self.assertGreater(point.physics_step, point.snapshot_physics_step)
            self.assertGreater(point.branch_chunk_index, 0)
            self.assertEqual(point.success_rollout_id, success.rollout_id)

    def test_select_branch_points_uses_consensus_over_all_failures(self):
        # Two failures with different divergence profiles: the selector must
        # use the MEAN divergence over all failures, not just the first one.
        chunk_sizes = (4, 4, 4, 4, 4)
        success_trace = _make_trace(chunk_sizes=chunk_sizes)
        # _make_trace only creates 3 snapshots; add a 4th at physics step 13
        # (mid chunk 3) so the causal index can land off zero.
        step_13 = success_trace["control_steps"][13]
        success_trace["branch_snapshots"].append(
            {
                "snapshot_id": 3,
                "physics_step": 13,
                "control_trace_offset": 13,
                "robot_state": step_13["robot_state"],
                "observation_joint_vector": np.asarray(
                    step_13["robot_state"]["joints"], dtype=np.float64
                ),
            }
        )

        def _failure(jump_step: float, jump: float) -> dict:
            trace = _make_trace(chunk_sizes=chunk_sizes)
            for step in trace["control_steps"]:
                offset = 0.1 if step["physics_step"] < jump_step else jump
                step["robot_state"]["joints"] = np.asarray(
                    [float(step["physics_step"]) + offset], dtype=np.float64
                )
            return trace

        # Snapshot steps are 1, 5, 9, 13.
        # failure_b (listed FIRST) diverges at step 9: d_b = [0.1, 0.1, 3, 3]
        #   -> alone it would put k* = 1 and the causal point at snapshot 0.
        # failure_a diverges at step 13:              d_a = [0.1, 0.1, 0.1, 8]
        # consensus mean = [0.1, 0.1, 1.55, 5.5], diffs [0, 1.45, 3.95]
        #   -> k* = 2, causal = snapshot 1.
        failure_b_trace = _failure(9, 3.0)
        failure_a_trace = _failure(13, 8.0)

        success = Candidate("task", 1, 0, True, Path("success.hdf5"))
        failures = [
            Candidate("task", 1, 1, False, Path("failure_b.hdf5")),
            Candidate("task", 1, 2, False, Path("failure_a.hdf5")),
        ]
        points = select_branch_points(
            "task",
            1,
            success,
            failures,
            max_points=2,
            rng=__import__("random").Random(0),
            success_trace=success_trace,
            failure_traces=[failure_b_trace, failure_a_trace],
        )

        causal = next(p for p in points if p.point_type == "pre_divergence_causal")
        self.assertEqual(causal.snapshot_id, 1)
        # Growth window {1, 2, 3} leaves snapshot 0 as the only control.
        control = next(p for p in points if p.point_type == "matched_random_control")
        self.assertEqual(control.snapshot_id, 0)

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
