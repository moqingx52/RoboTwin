"""Fixed-draw held-out anchor probe evaluation without polluting training RNG."""

from __future__ import annotations

from typing import Any

import torch


def _select_cfg_value(cfg, key: str, default):
    try:
        from omegaconf import OmegaConf
    except ImportError:
        return default
    return OmegaConf.select(cfg, key, default=default)


def _index_obs(obs, indices):
    return {key: value.index_select(0, indices) for key, value in obs.items()}


def materialize_probe_draws(
    teacher,
    batch: dict[str, Any],
    cfg,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    """Precompute teacher action, noise and timesteps once for a fixed probe batch."""
    preservation = batch["sample_preservation_group"]
    group_sources = {
        str(key): int(value)
        for key, value in dict(
            _select_cfg_value(cfg, "training.brace_anchor.groups", {"base_solved": 1, "boundary": 2})
        ).items()
    }
    max_per_group = int(_select_cfg_value(cfg, "training.brace_anchor.samples_per_group", 8))
    draws: dict[str, dict[str, torch.Tensor]] = {}
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    teacher.eval()
    with torch.no_grad():
        for group, group_id in group_sources.items():
            indices = torch.nonzero(preservation == group_id, as_tuple=False).flatten()[:max_per_group]
            if indices.numel() == 0:
                continue
            obs = _index_obs(batch["obs"], indices)
            clean_action = teacher.predict_action(obs)["action_pred"]
            noise = torch.randn(clean_action.shape, device=clean_action.device, dtype=clean_action.dtype, generator=generator)
            timesteps = torch.randint(
                0,
                teacher.noise_scheduler.config.num_train_timesteps,
                (clean_action.shape[0],),
                device=clean_action.device,
                generator=generator,
            ).long()
            noisy_action = teacher.make_noisy_action(clean_action, noise, timesteps)
            teacher_pred = teacher.denoise_action(obs, noisy_action, timesteps)
            draws[group] = {
                "indices": indices,
                "obs": obs,
                "noise": noise,
                "timesteps": timesteps,
                "noisy_action": noisy_action,
                "teacher_pred": teacher_pred.detach(),
            }
    return draws


def evaluate_probe_draws(
    student,
    teacher,
    draws: dict[str, dict[str, torch.Tensor]],
    cfg,
    *,
    reference_student=None,
) -> tuple[dict[str, float], dict[str, float]]:
    constraints: dict[str, float] = {}
    monitor: dict[str, float] = {}
    student_was_training = student.training
    student.eval()
    teacher.eval()
    if reference_student is not None:
        reference_student.eval()
    try:
        for group, payload in draws.items():
            obs = payload["obs"]
            noisy_action = payload["noisy_action"]
            timesteps = payload["timesteps"]
            teacher_pred = payload["teacher_pred"]
            student_pred = student.denoise_action(obs, noisy_action, timesteps)
            constraints[group] = float(torch.mean((student_pred - teacher_pred) ** 2).detach().item())
            if reference_student is not None:
                with torch.no_grad():
                    ref_pred = reference_student.denoise_action(obs, noisy_action, timesteps)
                    monitor[f"{group}_ema_drift"] = float(
                        torch.mean((ref_pred - teacher_pred) ** 2).detach().item()
                    )
    finally:
        student.train(student_was_training)
    return constraints, monitor
