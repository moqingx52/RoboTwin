import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from experiments.brace.replay_audit import (
    Candidate,
    checkpoint_indices,
    load_episode,
    pose_errors,
    quaternion_error,
    select_candidates,
)


class ReplayAuditTest(unittest.TestCase):
    def test_checkpoint_indices_are_unique_and_interior(self):
        self.assertEqual(checkpoint_indices(13, 3), [3, 6, 9])

    def test_quaternion_sign_is_equivalent(self):
        quaternion = np.array([0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(quaternion_error(quaternion, -quaternion), 0.0)
        translation, rotation = pose_errors(
            np.r_[np.zeros(3), quaternion],
            np.r_[np.zeros(3), -quaternion],
        )
        self.assertAlmostEqual(translation, 0.0)
        self.assertAlmostEqual(rotation, 0.0)

    def test_selection_contains_both_outcomes(self):
        candidates = [
            Candidate("task", index, index, index < 12, Path(f"{index}.hdf5"))
            for index in range(20)
        ]
        selected, errors = select_candidates(candidates, 20, seed=0, require_both=True)
        self.assertFalse(errors)
        self.assertEqual(len(selected), 20)
        self.assertTrue(any(item.success for item in selected))
        self.assertTrue(any(not item.success for item in selected))

    def test_hdf5_requires_task_object_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.hdf5"
            with h5py.File(path, "w") as root:
                root.require_group("joint_action").create_dataset("vector", data=np.zeros((4, 14)))
                endpose = root.require_group("endpose")
                endpose.create_dataset("left_endpose", data=np.zeros((4, 7)))
                endpose.create_dataset("right_endpose", data=np.zeros((4, 7)))
            with self.assertRaisesRegex(ValueError, "task_object_pose"):
                load_episode(path)

    def test_load_complete_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.hdf5"
            with h5py.File(path, "w") as root:
                root.require_group("joint_action").create_dataset("vector", data=np.zeros((4, 14)))
                endpose = root.require_group("endpose")
                endpose.create_dataset("left_endpose", data=np.zeros((4, 7)))
                endpose.create_dataset("right_endpose", data=np.zeros((4, 7)))
                root.require_group("task_object_pose").create_dataset(
                    "object", data=np.zeros((4, 7))
                )
            episode = load_episode(path)
            self.assertEqual(episode.length, 4)
            self.assertEqual(set(episode.objects), {"object"})


if __name__ == "__main__":
    unittest.main()
