import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

from experiments.brace.control_trace import (
    append_brace_trace_to_hdf5,
    load_brace_trace,
    snapshot_indices,
    validate_schema_v2,
)


class ControlTraceTest(unittest.TestCase):
    def test_snapshot_indices_are_unique_and_interior(self):
        self.assertEqual(snapshot_indices(13, 3), [3, 6, 9])

    def test_hdf5_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.hdf5"
            with h5py.File(path, "w") as root:
                root.create_dataset("placeholder", data=1)

            control_steps = []
            for step in range(5):
                control_steps.append(
                    {
                        "physics_step": step,
                        "policy_chunk_index": 0,
                        "left_arm_pos": np.zeros(6),
                        "left_arm_vel": np.zeros(6),
                        "right_arm_pos": np.zeros(6),
                        "right_arm_vel": np.zeros(6),
                        "left_gripper": 0.0,
                        "right_gripper": 0.0,
                        "robot_state": {
                            "left_qpos": np.zeros(7),
                            "left_qvel": np.zeros(7),
                            "right_qpos": np.zeros(7),
                            "right_qvel": np.zeros(7),
                            "joints": np.zeros(14),
                            "left_endpose": np.zeros(7),
                            "right_endpose": np.zeros(7),
                            "dynamic_actors": {
                                "container": {
                                    "pose": np.zeros(7),
                                    "linear_velocity": np.zeros(3),
                                    "angular_velocity": np.zeros(3),
                                }
                            },
                        },
                    }
                )

            branch_snapshots = [
                {
                    "snapshot_id": 0,
                    "physics_step": 2,
                    "control_trace_offset": 2,
                    "robot_state": control_steps[2]["robot_state"],
                    "observation_joint_vector": control_steps[2]["robot_state"]["joints"],
                }
            ]
            append_brace_trace_to_hdf5(
                path,
                policy_chunks=[
                    {
                        "chunk_index": 0,
                        "policy_step": 0,
                        "physics_step": 0,
                        "action": np.zeros(14),
                    }
                ],
                control_steps=control_steps,
                branch_snapshots=branch_snapshots,
                meta={"schema_version": 2, "physics_steps": 5},
            )
            self.assertEqual(validate_schema_v2(path), [])
            payload = load_brace_trace(path)
            self.assertEqual(len(payload["control_steps"]), 5)
            self.assertEqual(len(payload["branch_snapshots"]), 1)

    def test_replay_control_step_does_not_call_topp(self):
        from experiments.brace.control_trace import replay_control_step

        env = mock.Mock()
        step = {
            "left_arm_pos": np.zeros(6),
            "left_arm_vel": np.zeros(6),
            "right_arm_pos": np.zeros(6),
            "right_arm_vel": np.zeros(6),
            "left_gripper": 0.0,
            "right_gripper": 0.0,
        }
        replay_control_step(env, step)
        env.robot.set_arm_joints.assert_called()
        env.scene.step.assert_called_once()


if __name__ == "__main__":
    unittest.main()
