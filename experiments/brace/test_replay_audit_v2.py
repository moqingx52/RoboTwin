import unittest
from pathlib import Path
from unittest import mock

from experiments.brace.replay_audit import Candidate
from experiments.brace.replay_audit_v2 import (
    audit_candidates_parallel,
    job_worker_index,
    resolve_worker_count,
    run_audit_job,
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
                )


if __name__ == "__main__":
    unittest.main()
