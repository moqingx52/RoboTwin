#!/usr/bin/env python3
"""Trajectory-level KL (D_path) between a frozen teacher and a student DP.

Replaces the archived single-timestep denoising-MSE anchor. For the actual
deployment reverse process (num_inference_steps=100, variance_type=fixed_small,
clip_sample=True), both policies define reverse Gaussians at every step t:

    p_theta(x_{t-1} | x_t) = N(mu_theta(x_t, t), sigma_t^2 I)

with the SAME sigma_t (fixed_small: variance comes from the scheduler alone).
Along a shared trajectory x_T..x_0, the per-step KL is exactly

    KL_t = || mu_teacher(x_t,t) - mu_student(x_t,t) ||^2 / (2 sigma_t^2)

and D_path = sum_t KL_t (expectation over the teacher's path measure) upper
bounds the marginal action-distribution KL by the data-processing inequality:

    KL( pi_teacher(a|s) || pi_student(a|s) ) <= D_path(s)
    | E_{pi_student} Q - E_{pi_teacher} Q |  <= sqrt( D_path / 2 )     (Pinsker)

Because clip_sample=True makes the reverse mean a NONLINEAR function of the
model output (x0 prediction is clamped to [-1, 1] before recombination), we
compute the actual reverse means by mirroring DDPMScheduler.step, rather than
using the epsilon-difference closed form (which is only valid without
clipping).

Two entry points:
  - path_kl_terms(...): full-schedule evaluation along the teacher's
    deployment trajectory (metric / probe eval; no grad).
  - path_kl_training_loss(...): differentiable stochastic estimate on a
    random subset of steps (training-time constraint on the student).
"""

from __future__ import annotations

import math
from typing import Any

import torch


def _extract(values: torch.Tensor, t: int) -> torch.Tensor:
    return values[t] if values.dim() > 0 else values


def reverse_mean_and_variance(
    scheduler,
    model_output: torch.Tensor,
    t: int,
    sample: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Posterior mean/variance of one reverse step, mirroring DDPMScheduler.step.

    Handles prediction_type in {epsilon, sample}, clip_sample, and
    variance_type=fixed_small — i.e. exactly the deployment configuration.
    Returns (mean, variance) where variance is the scalar sigma_t^2.
    """
    alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
    t = int(t)
    prev_t = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps

    alpha_prod_t = alphas_cumprod[t]
    alpha_prod_t_prev = alphas_cumprod[prev_t] if prev_t >= 0 else torch.ones_like(alpha_prod_t)
    beta_prod_t = 1.0 - alpha_prod_t
    beta_prod_t_prev = 1.0 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1.0 - current_alpha_t

    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
    elif prediction_type == "sample":
        pred_original_sample = model_output
    else:
        raise ValueError(f"unsupported prediction_type: {prediction_type}")

    if getattr(scheduler.config, "clip_sample", False):
        clip_range = getattr(scheduler.config, "clip_sample_range", 1.0)
        pred_original_sample = pred_original_sample.clamp(-clip_range, clip_range)

    pred_original_coeff = (alpha_prod_t_prev.sqrt() * current_beta_t) / beta_prod_t
    current_sample_coeff = (current_alpha_t.sqrt() * beta_prod_t_prev) / beta_prod_t
    mean = pred_original_coeff * pred_original_sample + current_sample_coeff * sample

    variance_type = getattr(scheduler.config, "variance_type", "fixed_small")
    if variance_type != "fixed_small":
        raise ValueError(
            f"D_path assumes variance_type=fixed_small (deployment config); got {variance_type}"
        )
    variance = float((beta_prod_t_prev / beta_prod_t * current_beta_t).clamp(min=1e-20))
    return mean, variance


@torch.no_grad()
def path_kl_terms(
    teacher_policy,
    student_policy,
    obs_dict: dict[str, torch.Tensor],
    *,
    generator: torch.Generator | None = None,
    num_inference_steps: int | None = None,
) -> dict[str, Any]:
    """Run the teacher's deployment reverse process; accumulate per-step KL.

    Both policies see the identical x_t at every step (the teacher's
    trajectory), so this evaluates E_{teacher path}[ sum_t KL_t ] by a single
    path sample — average over obs/generator draws for the expectation.

    Returns per-sample D_path [B], the per-step KL matrix [B, S], the sampled
    action difference, and the Pinsker bound sqrt(D_path/2).
    """
    scheduler = teacher_policy.noise_scheduler
    steps = num_inference_steps or teacher_policy.num_inference_steps

    t_local, t_global = teacher_policy._observation_condition(obs_dict)
    s_local, s_global = student_policy._observation_condition(obs_dict)

    batch = next(iter(obs_dict.values())).shape[0]
    device = next(iter(obs_dict.values())).device
    dtype = teacher_policy.normalizer["action"].params_dict["offset"].dtype
    shape = (batch, teacher_policy.horizon, teacher_policy.action_dim)

    trajectory = torch.randn(size=shape, dtype=dtype, device=device, generator=generator)
    scheduler.set_timesteps(steps)

    kl_steps = []
    for t in scheduler.timesteps:
        teacher_out = teacher_policy.model(trajectory, t, local_cond=t_local, global_cond=t_global)
        student_out = student_policy.model(trajectory, t, local_cond=s_local, global_cond=s_global)
        teacher_mean, variance = reverse_mean_and_variance(scheduler, teacher_out, t, trajectory)
        student_mean, _ = reverse_mean_and_variance(scheduler, student_out, t, trajectory)
        kl_t = (teacher_mean - student_mean).pow(2).flatten(1).sum(dim=1) / (2.0 * variance)
        kl_steps.append(kl_t)
        # advance along the teacher's actual deployment path
        trajectory = scheduler.step(teacher_out, t, trajectory, generator=generator).prev_sample

    kl_matrix = torch.stack(kl_steps, dim=1)  # [B, S]
    d_path = kl_matrix.sum(dim=1)  # [B]
    return {
        "d_path": d_path,
        "kl_per_step": kl_matrix,
        "pinsker_bound": (d_path / 2.0).clamp(min=0).sqrt(),
        "final_trajectory": trajectory,
        "num_inference_steps": int(steps),
    }


def path_kl_training_loss(
    teacher_policy,
    student_policy,
    obs_dict: dict[str, torch.Tensor],
    *,
    n_timesteps: int = 4,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Differentiable stochastic estimate of D_path for the training constraint.

    Instead of running the full 100-step loop per batch, draw x_t from the
    teacher's FORWARD process at n_timesteps random deployment steps (the
    forward marginal q(x_t | x_0) with x_0 sampled once from the teacher is a
    proper reparameterization of the teacher's path marginals for DDPM), and
    penalize the reverse-mean gap at those steps, importance-scaled so the
    expectation matches sum_t KL_t.

    Teacher runs under no_grad; gradients flow through the student only.
    """
    scheduler = teacher_policy.noise_scheduler
    steps = teacher_policy.num_inference_steps
    scheduler.set_timesteps(steps)
    timesteps_all = scheduler.timesteps

    with torch.no_grad():
        t_local, t_global = teacher_policy._observation_condition(obs_dict)
        batch = next(iter(obs_dict.values())).shape[0]
        device = next(iter(obs_dict.values())).device
        dtype = teacher_policy.normalizer["action"].params_dict["offset"].dtype
        shape = (batch, teacher_policy.horizon, teacher_policy.action_dim)
        # one teacher deployment sample per obs -> x_0 ~ pi_teacher
        x0 = teacher_policy.conditional_sample(
            condition_data=torch.zeros(shape, dtype=dtype, device=device),
            condition_mask=torch.zeros(shape, dtype=torch.bool, device=device),
            local_cond=t_local,
            global_cond=t_global,
            generator=generator,
        )

    s_local, s_global = student_policy._observation_condition(obs_dict)

    step_indices = torch.randperm(len(timesteps_all), generator=None)[:n_timesteps]
    scale = float(len(timesteps_all)) / float(len(step_indices))
    alphas_cumprod = scheduler.alphas_cumprod.to(device=x0.device, dtype=x0.dtype)

    total = x0.new_zeros(batch)
    for idx in step_indices.tolist():
        t = int(timesteps_all[idx])
        noise = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, generator=generator)
        a_bar = alphas_cumprod[t]
        x_t = a_bar.sqrt() * x0 + (1.0 - a_bar).sqrt() * noise
        with torch.no_grad():
            teacher_out = teacher_policy.model(x_t, t, local_cond=t_local, global_cond=t_global)
            teacher_mean, variance = reverse_mean_and_variance(scheduler, teacher_out, t, x_t)
        student_out = student_policy.model(x_t, t, local_cond=s_local, global_cond=s_global)
        student_mean, _ = reverse_mean_and_variance(scheduler, student_out, t, x_t)
        total = total + (teacher_mean - student_mean).pow(2).flatten(1).sum(dim=1) / (2.0 * variance)

    return scale * total.mean()


def summarize_path_kl(d_path_values: torch.Tensor) -> dict[str, float]:
    values = d_path_values.detach().float().cpu()
    return {
        "mean_d_path": float(values.mean()),
        "median_d_path": float(values.median()),
        "p95_d_path": float(values.quantile(0.95)) if len(values) > 1 else float(values.max()),
        "max_d_path": float(values.max()),
        "mean_pinsker_bound": float((values / 2.0).clamp(min=0).sqrt().mean()),
        "samples": int(len(values)),
    }
