#!/usr/bin/env python3
"""Semantic regression tests for BRACE source vs preservation-group separation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

try:
    import zarr
except ImportError:  # pragma: no cover
    zarr = None

REPO_ROOT = Path(__file__).resolve().parents[2]

from experiments.brace.preservation_groups import (
    PRESERVATION_BASE_SOLVED,
    PRESERVATION_BOUNDARY,
    PRESERVATION_NONE,
    SOURCE_EXPERT,
    SOURCE_ROLLOUT,
)


def _write_minimal_zarr(path: Path, *, sources, preservation, env_seeds):
    root = zarr.group(str(path))
    data = root.create_group("data")
    horizon = 2
    total = len(sources) * horizon
    data.create_dataset("head_camera", shape=(total, 3, 4, 4), dtype=np.uint8)
    data.create_dataset("state", shape=(total, 2, 4), dtype=np.float32)
    data.create_dataset("action", shape=(total, 2, 4), dtype=np.float32)
    meta = root.create_group("meta")
    ends = horizon * np.arange(1, len(sources) + 1, dtype=np.int64)
    meta.create_dataset("episode_ends", data=ends, dtype="int64")
    meta.create_dataset("episode_source", data=np.asarray(sources, dtype=np.int64), dtype="int64")
    meta.create_dataset("episode_preservation_group", data=np.asarray(preservation, dtype=np.int64), dtype="int64")
    meta.create_dataset("episode_env_seed", data=np.asarray(env_seeds, dtype=np.int64), dtype="int64")


class PreservationSemanticsTests(unittest.TestCase):
    def test_expert_source_is_not_base_solved_group(self):
        self.assertNotEqual(SOURCE_EXPERT, PRESERVATION_BASE_SOLVED)
        self.assertNotEqual(SOURCE_ROLLOUT, PRESERVATION_BOUNDARY)

    @unittest.skipIf(zarr is None, "zarr not installed")
    def test_dataset_exposes_separate_batch_fields(self):
        from diffusion_policy.dataset.robot_image_dataset import RobotImageDataset

        with tempfile.TemporaryDirectory() as tmp:
            zarr_path = Path(tmp) / "data.zarr"
            _write_minimal_zarr(
                zarr_path,
                sources=[SOURCE_EXPERT, SOURCE_ROLLOUT],
                preservation=[PRESERVATION_BASE_SOLVED, PRESERVATION_BOUNDARY],
                env_seeds=[-1, 100001],
            )
            dataset = RobotImageDataset(str(zarr_path), batch_size=2, horizon=2, pad_before=0, pad_after=0)
            batch = dataset[np.asarray([0, 1], dtype=np.int64)]
            self.assertIn("sample_source", batch)
            self.assertIn("sample_preservation_group", batch)
            self.assertIn("sample_env_seed", batch)
            self.assertNotEqual(int(batch["sample_source"][0]), int(batch["sample_preservation_group"][0]))

    def test_anchor_requires_preservation_group(self):
        from diffusion_policy.workspace.robotworkspace import BraceDualState, compute_brace_anchor_loss

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Linear(1, 1)

            def denoise_action(self, obs, noisy, timesteps):
                return noisy

        class DummyTeacher(DummyModel):
            noise_scheduler = type("S", (), {"config": type("C", (), {"num_train_timesteps": 4})()})()

            def predict_action(self, obs):
                return {"action_pred": obs["agent_pos"]}

            def make_noisy_action(self, clean, noise, timesteps):
                return clean + noise

        student = DummyModel()
        teacher = DummyTeacher()
        dual = BraceDualState(["base_solved", "boundary"])
        cfg = OmegaConf.create(
            {
                "training": {
                    "brace_anchor": {
                        "groups": {"base_solved": 1, "boundary": 2},
                        "epsilon": 1e-4,
                        "samples_per_group": 8,
                    }
                }
            }
        )
        with self.assertRaises(ValueError):
            compute_brace_anchor_loss(student, teacher, {"obs": {"agent_pos": torch.randn(2, 2, 1)}}, cfg, dual)

    def test_screen_dataset_writes_none_preservation_for_chunks(self):
        from experiments.brace.build_screen_dataset import PRESERVATION_NONE

        self.assertEqual(PRESERVATION_NONE, 0)

    def test_missing_preservation_group_fails_closed_for_anchor_enabled_training(self):
        cfg = OmegaConf.create(
            {
                "training": {
                    "brace_anchor": {
                        "enabled": True,
                        "dataset": {"zarr_path": None},
                        "groups": {"base_solved": 1, "boundary": 2},
                    }
                },
                "task": {"dataset": {"zarr_path": "/tmp/unused"}},
                "dataloader": {"batch_size": 2, "shuffle": False, "num_workers": 0},
            }
        )
        from diffusion_policy.workspace.robotworkspace import compute_brace_anchor_loss, BraceDualState

        class Dummy(torch.nn.Module):
            noise_scheduler = type("S", (), {"config": type("C", (), {"num_train_timesteps": 4})()})()

            def denoise_action(self, obs, noisy, timesteps):
                return noisy

            def predict_action(self, obs):
                return {"action_pred": obs["agent_pos"]}

            def make_noisy_action(self, clean, noise, timesteps):
                return clean + noise

        with self.assertRaises(ValueError):
            compute_brace_anchor_loss(
                Dummy(),
                Dummy(),
                {"obs": {"agent_pos": torch.randn(2, 2, 1)}},
                cfg,
                BraceDualState(["base_solved", "boundary"]),
            )


if __name__ == "__main__":
    unittest.main()
