import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

from experiments.brace.control_trace import (
    append_brace_trace_to_hdf5,
    capture_model_obs_history,
    chunk_boundary_snapshot_indices,
    load_brace_trace,
    restore_model_obs_history,
    snapshot_indices,
    validate_schema_v2,
)


class _FakeRunner:
    def __init__(self, n_obs_steps: int = 3):
        self.n_obs_steps = n_obs_steps
        self.obs = deque(maxlen=n_obs_steps + 1)


class _FakeModel:
    """Mimics the DP wrapper's obs-deque interface (runner/update_obs/reset_obs)."""

    def __init__(self, n_obs_steps: int = 3):
        self.runner = _FakeRunner(n_obs_steps)

    def update_obs(self, obs):
        self.runner.obs.append(obs)

    def reset_obs(self):
        self.runner.obs.clear()


def _make_obs_frame(value: int) -> dict:
    def cam(v: int) -> np.ndarray:
        return np.full((3, 4, 4), v, dtype=np.uint8).astype(np.float32) / 255.0

    return {
        "head_cam": cam(value),
        "left_cam": cam(value + 1),
        "right_cam": cam(value + 2),
        "agent_pos": np.full(14, float(value), dtype=np.float64),
    }


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

            obs_history = {
                "head_cam": np.stack([np.full((3, 4, 4), k, dtype=np.uint8) for k in (10, 20, 30)]).astype(np.float32) / 255.0,
                "left_cam": np.stack([np.full((3, 4, 4), k, dtype=np.uint8) for k in (40, 50, 60)]).astype(np.float32) / 255.0,
                "right_cam": np.stack([np.full((3, 4, 4), k, dtype=np.uint8) for k in (70, 80, 90)]).astype(np.float32) / 255.0,
                "agent_pos": np.arange(3 * 14, dtype=np.float64).reshape(3, 14),
            }
            branch_snapshots = [
                {
                    "snapshot_id": 0,
                    "physics_step": 2,
                    "control_trace_offset": 2,
                    "robot_state": control_steps[2]["robot_state"],
                    "observation_joint_vector": control_steps[2]["robot_state"]["joints"],
                    "obs_history": obs_history,
                }
            ]
            append_brace_trace_to_hdf5(
                path,
                policy_chunks=[
                    {
                        "chunk_index": 0,
                        "policy_step": 0,
                        "physics_step": 0,
                        "action": np.zeros((6, 14)),
                    }
                ],
                control_steps=control_steps,
                branch_snapshots=branch_snapshots,
                meta={"schema_version": 2, "physics_steps": 5, "n_action_steps": 6},
            )
            self.assertEqual(validate_schema_v2(path), [])
            payload = load_brace_trace(path)
            self.assertEqual(len(payload["control_steps"]), 5)
            self.assertEqual(len(payload["branch_snapshots"]), 1)
            self.assertEqual(payload["policy_chunks"][0]["action"].shape, (6, 14))
            loaded_history = payload["branch_snapshots"][0]["obs_history"]
            self.assertIsNotNone(loaded_history)
            # k/255 float32 cams survive the uint8 quantization exactly.
            for cam_key in ("head_cam", "left_cam", "right_cam"):
                np.testing.assert_array_equal(loaded_history[cam_key], obs_history[cam_key])
                self.assertEqual(loaded_history[cam_key].dtype, np.float32)
            np.testing.assert_array_equal(loaded_history["agent_pos"], obs_history["agent_pos"])

    def test_legacy_single_step_policy_chunk_still_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.hdf5"
            with h5py.File(path, "w") as root:
                root.create_dataset("placeholder", data=1)
            control_steps = [
                {
                    "physics_step": 0,
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
                branch_snapshots=[],
                meta={"schema_version": 2, "physics_steps": 1, "n_action_steps": 1},
            )
            payload = load_brace_trace(path)
            self.assertEqual(payload["policy_chunks"][0]["action"].shape, (1, 14))

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


class ObsHistoryTest(unittest.TestCase):
    def test_chunk_boundary_snapshot_indices(self):
        control_steps = [
            {"policy_chunk_index": 0},
            {"policy_chunk_index": 0},
            {"policy_chunk_index": 1},
            {"policy_chunk_index": 1},
            {"policy_chunk_index": 1},
            {"policy_chunk_index": 2},
            {"policy_chunk_index": 2},
        ]
        # Last control step of each chunk except the final chunk.
        self.assertEqual(chunk_boundary_snapshot_indices(control_steps), [1, 4])

    def test_capture_restore_history_parity(self):
        source = _FakeModel(n_obs_steps=3)
        frames = [_make_obs_frame(v) for v in (10, 20, 30)]
        for frame in frames:
            source.update_obs(frame)

        captured = capture_model_obs_history(source)
        self.assertIsNotNone(captured)
        self.assertEqual(captured["agent_pos"].shape, (3, 14))

        # Round-trip through HDF5 (uint8 cam quantization must be lossless
        # for k/255 inputs).
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parity.hdf5"
            with h5py.File(path, "w") as root:
                root.create_dataset("placeholder", data=1)
            snapshot = {
                "snapshot_id": 0,
                "physics_step": 0,
                "control_trace_offset": 0,
                "robot_state": {
                    "left_qpos": np.zeros(7),
                    "left_qvel": np.zeros(7),
                    "right_qpos": np.zeros(7),
                    "right_qvel": np.zeros(7),
                    "joints": np.zeros(14),
                    "left_endpose": np.zeros(7),
                    "right_endpose": np.zeros(7),
                    "dynamic_actors": {},
                },
                "observation_joint_vector": np.zeros(14),
                "obs_history": captured,
            }
            control_steps = [
                {
                    "physics_step": 0,
                    "policy_chunk_index": 0,
                    "left_arm_pos": np.zeros(6),
                    "left_arm_vel": np.zeros(6),
                    "right_arm_pos": np.zeros(6),
                    "right_arm_vel": np.zeros(6),
                    "left_gripper": 0.0,
                    "right_gripper": 0.0,
                    "robot_state": snapshot["robot_state"],
                }
            ]
            append_brace_trace_to_hdf5(
                path,
                policy_chunks=[
                    {"chunk_index": 0, "policy_step": 0, "physics_step": 0, "action": np.zeros((1, 14))}
                ],
                control_steps=control_steps,
                branch_snapshots=[snapshot],
                meta={"schema_version": 2, "physics_steps": 1, "n_action_steps": 1},
            )
            loaded_history = load_brace_trace(path)["branch_snapshots"][0]["obs_history"]

        target = _FakeModel(n_obs_steps=3)
        target.update_obs(_make_obs_frame(99))  # stale obs must be cleared
        boundary_frame = restore_model_obs_history(target, loaded_history)

        # reset_obs + update_obs(frames[:-1]); the newest frame is returned
        # for the caller to feed to get_action/update_obs.
        self.assertIsNotNone(boundary_frame)
        self.assertEqual(len(target.runner.obs), 2)
        restored = list(target.runner.obs) + [boundary_frame]
        for original, roundtripped in zip(frames, restored):
            for cam_key in ("head_cam", "left_cam", "right_cam"):
                np.testing.assert_array_equal(roundtripped[cam_key], original[cam_key])
            np.testing.assert_array_equal(roundtripped["agent_pos"], original["agent_pos"])

    def test_capture_repeat_fills_short_history(self):
        # After a reset the DP runner repeat-fills missing frames with the
        # earliest observation; capture must mirror that so restore is exact.
        model = _FakeModel(n_obs_steps=3)
        only_frame = _make_obs_frame(7)
        model.update_obs(only_frame)
        captured = capture_model_obs_history(model)
        self.assertIsNotNone(captured)
        self.assertEqual(captured["agent_pos"].shape, (3, 14))
        for row in range(3):
            np.testing.assert_array_equal(captured["agent_pos"][row], only_frame["agent_pos"])
            np.testing.assert_array_equal(captured["head_cam"][row], only_frame["head_cam"])

    def test_capture_keeps_only_last_n_obs_steps(self):
        model = _FakeModel(n_obs_steps=3)
        frames = [_make_obs_frame(v) for v in (1, 2, 3, 4)]  # deque maxlen is 4
        for frame in frames:
            model.update_obs(frame)
        captured = capture_model_obs_history(model)
        np.testing.assert_array_equal(
            captured["agent_pos"], np.stack([f["agent_pos"] for f in frames[-3:]])
        )

    def test_capture_returns_none_without_runner_or_obs(self):
        self.assertIsNone(capture_model_obs_history(object()))
        self.assertIsNone(capture_model_obs_history(_FakeModel(n_obs_steps=3)))

    def test_restore_none_history_resets_and_returns_none(self):
        model = _FakeModel(n_obs_steps=3)
        model.update_obs(_make_obs_frame(5))
        self.assertIsNone(restore_model_obs_history(model, None))
        self.assertEqual(len(model.runner.obs), 0)


if __name__ == "__main__":
    unittest.main()
