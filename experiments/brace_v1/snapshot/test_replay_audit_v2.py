import unittest
from pathlib import Path
from unittest import mock

from experiments.brace.replay_audit import Candidate
from experiments.brace.replay_audit_v2 import (
    actor_metrics_for_task,
    apply_actor_metrics,
    audit_candidates_parallel,
    evaluate_replay_gate,
    evaluate_task_replay_gate,
    job_worker_index,
    replay_gate_requirements,
    resolve_worker_count,
    run_audit_job,
    summarize_per_actor_failures,
    worker_gpu_assignments,
)


class ReplayAuditV2SchedulerTest(unittest.TestCase):
    def test_worker_gpu_assignments_pin_three_workers_per_gpu(self):
        assignments = worker_gpu_assignments([0, 1], workers_per_gpu=3)
        self.assertEqual(
            assignments,
            [
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 1),
                (4, 1),
                (5, 1),
            ],
        )

    def test_job_worker_index_round_robin(self):
        self.assertEqual(job_worker_index(0, 6), 0)
        self.assertEqual(job_worker_index(5, 6), 5)
        self.assertEqual(job_worker_index(6, 6), 0)

    def test_resolve_worker_count_caps_to_work_size(self):
        self.assertEqual(
            resolve_worker_count(
                work_size=10,
                workers=24,
                workers_per_gpu=3,
                gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            10,
        )

    def test_run_audit_job_loads_trace_once(self):
        candidate = Candidate("task", 1, 2, True, Path("episode.hdf5"))
        trace = {
            "branch_snapshots": [
                {"snapshot_id": 0, "physics_step": 1, "robot_state": {}},
                {"snapshot_id": 1, "physics_step": 2, "robot_state": {}},
                {"snapshot_id": 2, "physics_step": 3, "robot_state": {}},
            ],
            "control_steps": [],
        }
        with mock.patch(
            "experiments.brace.replay_audit_v2.load_brace_trace",
            return_value=trace,
        ) as load_trace, mock.patch(
            "experiments.brace.replay_audit_v2.audit_candidate_snapshots",
            return_value=[{"passed": True}],
        ) as audit_snapshots:
            rows = run_audit_job(
                candidate,
                snapshot_count=3,
                thresholds={},
                restore_repeat_count=1,
                horizon=1,
                task_config="demo_brace_trace",
            )

        load_trace.assert_called_once_with(candidate.path)
        audit_snapshots.assert_called_once()
        self.assertEqual(rows, [{"passed": True}])

    def test_audit_candidates_parallel_raises_when_worker_crashes(self):
        work = [
            Candidate("task", index, index, True, Path(f"{index}.hdf5"))
            for index in range(2)
        ]

        class FakeProcess:
            def __init__(self, *, exitcode):
                self.name = "replay-audit-v2-gpu-0-worker-0"
                self.exitcode = exitcode
                self._alive = False

            def start(self):
                return None

            def join(self):
                return None

            def is_alive(self):
                return self._alive

            def terminate(self):
                return None

        class FakeQueue:
            def __init__(self):
                self.closed = False

            def put(self, _item):
                return None

            def get(self, timeout=None):
                raise __import__("queue").Empty

            def close(self):
                self.closed = True

        fake_context = mock.Mock()
        fake_context.Queue.side_effect = lambda: FakeQueue()
        fake_context.Process.side_effect = lambda **kwargs: FakeProcess(exitcode=1)

        with mock.patch(
            "experiments.brace.replay_audit_v2.multiprocessing.get_context",
            return_value=fake_context,
        ):
            with self.assertRaisesRegex(RuntimeError, "worker exited before returning"):
                audit_candidates_parallel(
                    work,
                    {},
                    workers=2,
                    gpu_ids=[0],
                    workers_per_gpu=2,
                    snapshot_count=3,
                    restore_repeat_count=1,
                    horizon=1,
                    task_config="demo_brace_trace",
                    task_metrics_by_task={},
                )


class ReplayAuditV2GateTest(unittest.TestCase):
    def _protocol(self) -> dict:
        return {
            "tasks": ["place_container_plate", "dump_bin_bigbin"],
            "replay_gate": {
                "minimum_pass_rate": 0.95,
                "restore_determinism_pass_rate": 1.0,
                "control_trace_replay_minimum_pass_rate": 0.95,
                "per_task_control_trace_replay_minimum_pass_rate": 0.95,
            },
        }

    def _task_checks(self, task: str, replay_passed: int) -> list[dict]:
        checks: list[dict] = []
        for _ in range(60):
            checks.append({"task": task, "check_type": "restore_determinism", "passed": True})
        for _ in range(replay_passed):
            checks.append(
                {
                    "task": task,
                    "check_type": "control_trace_replay",
                    "passed": True,
                    "errors": {"object_rotation_error": 0.01},
                }
            )
        for _ in range(60 - replay_passed):
            checks.append(
                {
                    "task": task,
                    "check_type": "control_trace_replay",
                    "passed": False,
                    "errors": {"object_rotation_error": 0.1},
                }
            )
        return checks

    def test_mixed_pass_rate_masks_replay_failure(self):
        checks = self._task_checks("place_container_plate", 59) + self._task_checks("dump_bin_bigbin", 47)
        gate = evaluate_replay_gate(
            checks,
            self._protocol()["tasks"],
            self._protocol(),
            complete=True,
            preflight_errors=[],
        )
        self.assertAlmostEqual(gate["mixed_pass_rate"], 226 / 240)
        self.assertEqual(gate["control_trace_replay"]["passed_checks"], 106)
        self.assertFalse(gate["passed"])

    def test_per_task_gate_place_passes_dump_fails(self):
        checks = self._task_checks("place_container_plate", 59) + self._task_checks("dump_bin_bigbin", 47)
        requirements = replay_gate_requirements(self._protocol())
        place = evaluate_task_replay_gate(checks, "place_container_plate", requirements)
        dump = evaluate_task_replay_gate(checks, "dump_bin_bigbin", requirements)
        self.assertTrue(place["replay_gate_passed"])
        self.assertFalse(dump["replay_gate_passed"])
        self.assertEqual(place["control_trace_replay"]["passed_checks"], 59)
        self.assertEqual(dump["control_trace_replay"]["passed_checks"], 47)

    def test_actor_metrics_ignore_sphere_rotation(self):
        protocol = {
            **self._protocol(),
            "protocol_revision": "2.2",
            "actor_metrics": {
                "dump_bin_bigbin": {
                    "deskbin": ["translation", "rotation"],
                    "garbage_*": ["translation"],
                }
            },
        }
        errors = {
            "joint_max_error": 0.0,
            "end_effector_translation_error": 0.0,
            "end_effector_rotation_error": 0.0,
            "object_translation_error": 0.001,
            "object_rotation_error": 0.12,
            "actor_errors": {
                "deskbin": {"translation_error": 0.001, "rotation_error": 0.01},
                "garbage_0": {"translation_error": 0.001, "rotation_error": 0.12},
            },
        }
        result = apply_actor_metrics(
            errors,
            {
                "joint_max_error": 0.01,
                "end_effector_translation_error": 0.005,
                "end_effector_rotation_error": 0.05,
                "object_translation_error": 0.005,
                "object_rotation_error": 0.05,
            },
            actor_metrics_for_task(protocol, "dump_bin_bigbin"),
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["worst_actor"], None)

    def test_summarize_per_actor_failures_counts_garbage_rotation(self):
        checks = [
            {
                "task": "dump_bin_bigbin",
                "check_type": "control_trace_replay",
                "passed": False,
                "errors": {
                    "actor_errors": {
                        "garbage_0": {"translation_error": 0.0, "rotation_error": 0.1},
                    }
                },
                "actor_metric_passed": {
                    "garbage_0": {"translation": True, "rotation": False},
                },
            }
        ]
        counts = summarize_per_actor_failures(checks)
        self.assertEqual(counts["dump_bin_bigbin"]["garbage_0:rotation"], 1)


if __name__ == "__main__":
    unittest.main()
