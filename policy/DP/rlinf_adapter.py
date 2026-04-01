"""Stable observation helpers for RLinf rollout and RoboTwin DP (single head camera)."""

from __future__ import annotations

import numpy as np
import torch


def encode_observation_dict(observation: dict) -> dict:
    """RoboTwin simulator observation -> one-step DP dict (numpy, CHW float in [0, 1])."""
    if "observation" in observation:
        obs = observation["observation"]
        head = obs["head_camera"]["rgb"]
        head_cam = (np.moveaxis(np.asarray(head), -1, 0) / 255.0).astype(
            np.float32
        )
        agent_pos = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        return {"head_cam": head_cam, "agent_pos": agent_pos}

    head_cam = observation["head_cam"]
    agent_pos = observation["agent_pos"]
    if isinstance(head_cam, torch.Tensor):
        head_cam = head_cam.detach().cpu().numpy()
    if isinstance(agent_pos, torch.Tensor):
        agent_pos = agent_pos.detach().cpu().numpy()
    return {
        "head_cam": np.asarray(head_cam, dtype=np.float32),
        "agent_pos": np.asarray(agent_pos, dtype=np.float32),
    }


def rlinf_main_state_to_dp_timestep(
    main_images_bhwc: torch.Tensor,
    states_bd: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """RLinf RoboTwin env obs -> single-timestep tensors for DP history (B, C, H, W) / (B, D)."""
    if main_images_bhwc.dtype == torch.uint8:
        img = main_images_bhwc.float().div(255.0)
    else:
        img = main_images_bhwc.float()
        if img.max() > 1.5:
            img = img.div(255.0)
    img = img.permute(0, 3, 1, 2).contiguous()
    return {"head_cam": img, "agent_pos": states_bd.float().contiguous()}
