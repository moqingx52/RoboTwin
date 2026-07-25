import numpy as np
import torch
import hydra
import dill
import sys, os

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
sys.path.append(parent_dir)

from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.env_runner.dp_runner import DPRunner

class DP:

    def __init__(self, ckpt_file: str, n_obs_steps, n_action_steps, normalizer_zarr_path=None):
        self.policy = self.get_policy(ckpt_file, None, "cuda:0", normalizer_zarr_path=normalizer_zarr_path)
        self.runner = DPRunner(n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)

    def update_obs(self, observation):
        self.runner.update_obs(observation)
    
    def reset_obs(self):
        self.runner.reset_obs()

    def set_generator(self, generator):
        self.runner.generator = generator

    def get_action(self, observation=None):
        action = self.runner.get_action(self.policy, observation)
        return action

    def get_last_obs(self):
        return self.runner.obs[-1]

    def get_policy(self, checkpoint, output_dir, device, normalizer_zarr_path=None):
        # load checkpoint
        payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=output_dir)
        workspace: RobotWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        if normalizer_zarr_path is not None:
            from diffusion_policy.dataset.robot_image_dataset import RobotImageDataset

            dataset = RobotImageDataset(
                zarr_path=str(normalizer_zarr_path),
                horizon=cfg.horizon,
                pad_before=cfg.n_obs_steps - 1,
                pad_after=cfg.n_action_steps - 1,
                load_to_memory=False,
            )
            policy.set_normalizer(dataset.get_normalizer())

        device = torch.device(device)
        policy.to(device)
        policy.eval()

        return policy
