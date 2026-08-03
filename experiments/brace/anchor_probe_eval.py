"""Fixed-draw held-out anchor probe evaluation without polluting training RNG."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _select_cfg_value(cfg, key: str, default):
    try:
        from omegaconf import OmegaConf
    except ImportError:
        return default
    return OmegaConf.select(cfg, key, default=default)


def _index_obs(obs, indices):
    return {key: value.index_select(0, indices) for key, value in obs.items()}


def _materialize_single_draw(
    teacher,
    batch: dict[str, Any],
    cfg,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
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
            noise = torch.randn(
                clean_action.shape,
                device=clean_action.device,
                dtype=clean_action.dtype,
                generator=generator,
            )
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


def materialize_probe_draws(
    teacher,
    batch: dict[str, Any],
    cfg,
    *,
    seed: int,
    device: torch.device,
    num_draws: int = 1,
) -> list[dict[str, dict[str, torch.Tensor]]]:
    """Precompute fixed probe draws; fork_rng isolates teacher RNG side effects."""
    draws_list: list[dict[str, dict[str, torch.Tensor]]] = []
    fork_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        for draw_idx in range(max(1, int(num_draws))):
            draw_seed = int(seed) + draw_idx * 10007
            draws_list.append(
                _materialize_single_draw(teacher, batch, cfg, seed=draw_seed, device=device)
            )
    return draws_list


def _evaluate_single_draw(
    student,
    teacher,
    draws: dict[str, dict[str, torch.Tensor]],
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


def evaluate_probe_draws(
    student,
    teacher,
    draws_list: list[dict[str, dict[str, torch.Tensor]]],
    cfg,
    *,
    reference_student=None,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    """Evaluate one or more fixed draws; aggregate mean/p90/max across draws."""
    per_draw_raw: list[dict[str, float]] = []
    per_draw_monitor: list[dict[str, float]] = []
    for draws in draws_list:
        raw, monitor = _evaluate_single_draw(
            student,
            teacher,
            draws,
            reference_student=reference_student,
        )
        per_draw_raw.append(raw)
        per_draw_monitor.append(monitor)

    groups = sorted({group for draw in per_draw_raw for group in draw})
    aggregated_raw: dict[str, float] = {}
    aggregated_monitor: dict[str, float] = {}
    stats: dict[str, Any] = {"probe_draw_count": len(draws_list), "per_draw": per_draw_raw}

    for group in groups:
        values = [draw[group] for draw in per_draw_raw if group in draw]
        if values:
            arr = np.asarray(values, dtype=np.float64)
            aggregated_raw[group] = float(arr.mean())
            stats[f"probe_constraint_p90/{group}"] = float(np.percentile(arr, 90))
            stats[f"probe_constraint_max/{group}"] = float(arr.max())

    for key in sorted({k for draw in per_draw_monitor for k in draw}):
        values = [draw[key] for draw in per_draw_monitor if key in draw]
        if values:
            arr = np.asarray(values, dtype=np.float64)
            aggregated_monitor[key] = float(arr.mean())
            stats[f"probe_monitor_p90/{key}"] = float(np.percentile(arr, 90))

    return aggregated_raw, aggregated_monitor, stats
