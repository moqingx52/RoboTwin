import os
import sys

import dill
import hydra
import numpy as np
import torch

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
sys.path.append(parent_dir)

from diffusion_policy.env_runner.dp_runner import DPRunner
from diffusion_policy.workspace.robotworkspace import RobotWorkspace


def load_diffusion_policy_from_checkpoint(
    checkpoint: str,
    map_location="cpu",
):
    """Load a trained DiffusionUnetImagePolicy (+ normalizer state) from a RoboTwin DP .ckpt."""
    with open(checkpoint, "rb") as f:
        payload = torch.load(f, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=None)
    workspace: RobotWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    policy.to(torch.device(map_location))
    policy.eval()
    return policy


class DP:

    def __init__(self, ckpt_file: str, n_obs_steps, n_action_steps, device="cuda:0"):
        self.device = torch.device(device)
        self.policy = load_diffusion_policy_from_checkpoint(
            ckpt_file, map_location=str(self.device)
        )
        self.runner = DPRunner(n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)

    def update_obs(self, observation):
        self.runner.update_obs(observation)

    def reset_obs(self):
        self.runner.reset_obs()

    def get_action(
        self,
        observation=None,
        init_noise=None,
        return_chain=False,
    ):
        if return_chain:
            raise NotImplementedError("return_chain is not supported yet")
        return self.runner.get_action(
            self.policy, observation, init_noise=init_noise
        )

    def get_last_obs(self):
        return self.runner.obs[-1]

    def encode_obs(self, observation):
        """Map a generic env dict to DP obs keys (single head camera)."""
        from .rlinf_adapter import encode_observation_dict

        return encode_observation_dict(observation)

    def get_policy(self, checkpoint, output_dir, device):
        return load_diffusion_policy_from_checkpoint(
            checkpoint, map_location=device
        )
