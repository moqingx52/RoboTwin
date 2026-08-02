from typing import Dict
import numba
import torch
import numpy as np
import copy
import zarr
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
import pdb


class RobotImageDataset(BaseImageDataset):

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        batch_size=128,
        max_train_episodes=None,
        load_to_memory=True,
    ):

        super().__init__()
        zarr_root = zarr.open(zarr_path, mode="r")
        episode_ends = np.asarray(zarr_root["meta/episode_ends"][:], dtype=np.int64)
        if "episode_source" in zarr_root["meta"]:
            episode_sources = np.asarray(zarr_root["meta/episode_source"][:], dtype=np.int64)
            if episode_sources.shape != episode_ends.shape:
                raise ValueError(
                    f"episode_source shape {episode_sources.shape} does not match "
                    f"episode_ends shape {episode_ends.shape}"
                )
            frame_counts = np.diff(np.concatenate(([0], episode_ends)))
            self.frame_sources = np.repeat(episode_sources, frame_counts)
        else:
            self.frame_sources = None
        if "episode_env_seed" in zarr_root["meta"]:
            episode_env_seed = np.asarray(zarr_root["meta/episode_env_seed"][:], dtype=np.int64)
            if episode_env_seed.shape != episode_ends.shape:
                raise ValueError(
                    f"episode_env_seed shape {episode_env_seed.shape} does not match "
                    f"episode_ends shape {episode_ends.shape}"
                )
            frame_counts = np.diff(np.concatenate(([0], episode_ends)))
            self.frame_env_seed = np.repeat(episode_env_seed, frame_counts)
        else:
            self.frame_env_seed = None
        if "episode_preservation_group" in zarr_root["meta"]:
            episode_preservation = np.asarray(zarr_root["meta/episode_preservation_group"][:], dtype=np.int64)
            if episode_preservation.shape != episode_ends.shape:
                raise ValueError(
                    f"episode_preservation_group shape {episode_preservation.shape} does not match "
                    f"episode_ends shape {episode_ends.shape}"
                )
            frame_counts = np.diff(np.concatenate(([0], episode_ends)))
            self.frame_preservation_group = np.repeat(episode_preservation, frame_counts)
        else:
            self.frame_preservation_group = None
        replay_keys = ["head_camera", "state", "action"]
        if "sample_weight" in zarr_root["data"]:
            replay_keys.append("sample_weight")

        if load_to_memory:
            self.replay_buffer = ReplayBuffer.copy_from_path(
                zarr_path,
                # keys=['head_camera', 'front_camera', 'left_camera', 'right_camera', 'state', 'action'],
                keys=replay_keys,
            )
        else:
            print(f"Opening replay buffer on disk to limit host RAM: {zarr_path}")
            self.replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode="r")
        self.load_to_memory = bool(load_to_memory)

        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self._refresh_sample_sources()

        self.batch_size = batch_size
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        val_set._refresh_sample_sources()
        return val_set

    def _refresh_sample_sources(self):
        if self.frame_sources is None:
            self.sample_sources = None
            self.sample_env_seeds = None
            self.sample_preservation_groups = None
            self.sample_groups = None
            return
        # SequenceSampler indices are
        # (buffer_start, buffer_end, sample_start, sample_end). Every sequence
        # stays within one episode, so its first real frame identifies source.
        self.sample_sources = np.asarray(
            [self.frame_sources[int(row[0])] for row in self.sampler.indices],
            dtype=np.int64,
        )
        if self.frame_env_seed is not None:
            self.sample_env_seeds = np.asarray(
                [self.frame_env_seed[int(row[0])] for row in self.sampler.indices],
                dtype=np.int64,
            )
            # Backward-compatible alias used by rollout env-seed stratified sampling.
            self.sample_groups = self.sample_env_seeds
        else:
            self.sample_env_seeds = None
            self.sample_groups = None
        if self.frame_preservation_group is not None:
            self.sample_preservation_groups = np.asarray(
                [self.frame_preservation_group[int(row[0])] for row in self.sampler.indices],
                dtype=np.int64,
            )
        else:
            self.sample_preservation_groups = None

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["head_cam"] = get_image_range_normalizer()
        normalizer["front_cam"] = get_image_range_normalizer()
        normalizer["left_cam"] = get_image_range_normalizer()
        normalizer["right_cam"] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"].astype(np.float32)  # (agent_posx2, block_posex3)
        head_cam = np.moveaxis(sample["head_camera"], -1, 1) / 255
        # front_cam = np.moveaxis(sample['front_camera'],-1,1)/255
        # left_cam = np.moveaxis(sample['left_camera'],-1,1)/255
        # right_cam = np.moveaxis(sample['right_camera'],-1,1)/255

        data = {
            "obs": {
                "head_cam": head_cam,  # T, 3, H, W
                # 'front_cam': front_cam, # T, 3, H, W
                # 'left_cam': left_cam, # T, 3, H, W
                # 'right_cam': right_cam, # T, 3, H, W
                "agent_pos": agent_pos,  # T, D
            },
            "action": sample["action"].astype(np.float32),  # T, D
        }
        if "sample_weight" in sample:
            data["sample_weight"] = sample["sample_weight"].astype(np.float32)
        return data

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError  # Specialized
        elif isinstance(idx, int):
            sample = self.sampler.sample_sequence(idx)
            sample = dict_apply(sample, torch.from_numpy)
            return sample
        elif isinstance(idx, np.ndarray):
            assert len(idx) == self.batch_size
            if self.sample_sources is not None:
                self.buffers_torch["sample_source"] = torch.from_numpy(
                    self.sample_sources[idx].astype(np.int64, copy=False)
                )
            if self.sample_preservation_groups is not None:
                self.buffers_torch["sample_preservation_group"] = torch.from_numpy(
                    self.sample_preservation_groups[idx].astype(np.int64, copy=False)
                )
            if self.sample_env_seeds is not None:
                self.buffers_torch["sample_env_seed"] = torch.from_numpy(
                    self.sample_env_seeds[idx].astype(np.int64, copy=False)
                )
            for k, v in self.sampler.replay_buffer.items():
                if self.load_to_memory:
                    batch_sample_sequence(
                        self.buffers[k],
                        v,
                        self.sampler.indices,
                        idx,
                        self.sampler.sequence_length,
                    )
                else:
                    batch_sample_sequence_from_disk(
                        self.buffers[k],
                        v,
                        self.sampler.indices,
                        idx,
                        self.sampler.sequence_length,
                    )
            return self.buffers_torch
        else:
            raise ValueError(idx)

    def postprocess(self, samples, device):
        agent_pos = samples["state"].to(device, non_blocking=True)
        head_cam = samples["head_camera"].to(device, non_blocking=True) / 255.0
        # front_cam = samples['front_camera'].to(device, non_blocking=True) / 255.0
        # left_cam = samples['left_camera'].to(device, non_blocking=True) / 255.0
        # right_cam = samples['right_camera'].to(device, non_blocking=True) / 255.0
        action = samples["action"].to(device, non_blocking=True)
        data = {
            "obs": {
                "head_cam": head_cam,  # B, T, 3, H, W
                # 'front_cam': front_cam, # B, T, 3, H, W
                # 'left_cam': left_cam, # B, T, 3, H, W
                # 'right_cam': right_cam, # B, T, 3, H, W
                "agent_pos": agent_pos,  # B, T, D
            },
            "action": action,  # B, T, D
        }
        if "sample_weight" in samples:
            data["sample_weight"] = samples["sample_weight"].to(device, non_blocking=True).float()
        if "sample_source" in samples:
            data["sample_source"] = samples["sample_source"].to(device, non_blocking=True).long()
        if "sample_preservation_group" in samples:
            data["sample_preservation_group"] = samples["sample_preservation_group"].to(
                device, non_blocking=True
            ).long()
        if "sample_env_seed" in samples:
            data["sample_env_seed"] = samples["sample_env_seed"].to(device, non_blocking=True).long()
        return data


def _batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    for i in numba.prange(len(idx)):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[idx[i]]
        data[i, sample_start_idx:sample_end_idx] = input_arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = data[i, sample_start_idx]
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = data[i, sample_end_idx - 1]


_batch_sample_sequence_sequential = numba.jit(_batch_sample_sequence, nopython=True, parallel=False)
_batch_sample_sequence_parallel = numba.jit(_batch_sample_sequence, nopython=True, parallel=True)


def batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    batch_size = len(idx)
    assert data.shape == (batch_size, sequence_length, *input_arr.shape[1:])
    if batch_size >= 16 and data.nbytes // batch_size >= 2**16:
        _batch_sample_sequence_parallel(data, input_arr, indices, idx, sequence_length)
    else:
        _batch_sample_sequence_sequential(data, input_arr, indices, idx, sequence_length)


def batch_sample_sequence_from_disk(
    data: np.ndarray,
    input_arr,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    """Batch sampling path for zarr arrays that Numba cannot type directly."""
    batch_size = len(idx)
    assert data.shape == (batch_size, sequence_length, *input_arr.shape[1:])
    for i, sample_idx in enumerate(idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[sample_idx]
        sample = np.asarray(input_arr[buffer_start_idx:buffer_end_idx])
        data[i, sample_start_idx:sample_end_idx] = sample
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = sample[0]
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = sample[-1]
