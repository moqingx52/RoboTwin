"""Shared BRACE anchor diagnostic / calibration training loop."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

REPO_ROOT = Path(__file__).resolve().parents[2]

from experiments.brace.anchor_grad_diagnostics import (
    estimate_warmstart_lambdas,
    flatten_grad_norm,
    measure_basic_grad_norms,
    measure_checkpoint_grad_metrics,
)
from experiments.brace.anchor_probe_eval import evaluate_probe_draws, materialize_probe_draws
from experiments.brace.anchor_probe_split import (
    AnchorProbeSplit,
    build_anchor_probe_split,
    build_fixed_stratified_batch,
    sequence_indices_for_env_seeds,
)
from experiments.brace.anchor_training_smoke import (
    bootstrap_workspace,
    build_anchor_dataloader,
    build_sft_dataloader,
    build_training_config,
    ensure_anchor_zarr,
    ensure_screen_zarr,
    iter_postprocessed_batches,
    load_workspace,
)
from experiments.brace.replay_audit import git_commit, write_json_atomic
from experiments.brace.screen_gates import (
    evaluate_constraint_feasibility,
    summarize_feasibility_trajectory,
)


@dataclass
class DiagnosticJobConfig:
    job_id: str = "feasibility"
    anchor_enabled: bool = True
    diagnostic_steps: int = 1235
    scheduler_total_steps: int = 2470
    checkpoint_steps: list[int] = field(default_factory=lambda: [247, 741, 1235])
    checkpoint_epochs: dict[str, int] = field(default_factory=dict)
    dual_lr: float = 0.01
    lambda_max: float | None = None
    warmstart_target_ratio: float | None = None
    auto_fixed_beta_ratio: float | None = None
    grad_diag_at_checkpoints_only: bool = True
    record_peak_memory: bool = True
    early_stop_enabled: bool = True


def _anchor_env_seeds(batch: dict[str, Any]) -> list[int]:
    seeds = batch.get("sample_env_seed")
    if seeds is None:
        return []
    return [int(value) for value in seeds.detach().cpu().tolist()]


def apply_brace_anchor_overrides(cfg, job: DiagnosticJobConfig, brace_overrides: dict[str, Any] | None) -> None:
    brace = cfg.training.brace_anchor
    brace.enabled = bool(job.anchor_enabled)
    with open_dict(brace):
        for key, value in (brace_overrides or {}).items():
            if key == "lambda_init":
                continue
            if "." in key:
                OmegaConf.update(cfg, f"training.brace_anchor.{key}", value, merge=False)
            else:
                brace[key] = value
        if job.lambda_max is not None:
            brace.lambda_max = float(job.lambda_max)


def _epoch_for_step(step: int, checkpoint_epochs: dict[str, int], steps_per_epoch: int) -> int:
    if str(step) in checkpoint_epochs:
        return int(checkpoint_epochs[str(step)])
    return max(1, int(math.ceil(step / max(steps_per_epoch, 1))))


def _check_early_stop(
    *,
    row: dict[str, Any],
    state: dict[str, Any],
    job: DiagnosticJobConfig,
    lambda_max: float | None,
) -> str | None:
    if not job.early_stop_enabled:
        return None
    for key in ("train_loss", "total_loss"):
        value = row.get(key)
        if value is not None and (not math.isfinite(float(value))):
            return f"non_finite_{key}"
    for group_key in [k for k in row if k.startswith("brace_constraint/") or k.startswith("probe_constraint/")]:
        value = row.get(group_key)
        if value is not None and (not math.isfinite(float(value))):
            return f"non_finite_{group_key}"

    ratio = float(row.get("anchor_to_sft_grad_ratio", 0.0))
    if ratio > 5.0:
        state["high_grad_ratio_streak"] = int(state.get("high_grad_ratio_streak", 0)) + 1
    else:
        state["high_grad_ratio_streak"] = 0
    if state.get("high_grad_ratio_streak", 0) >= 20:
        return "grad_ratio_above_5x_for_20_steps"

    if lambda_max is not None:
        at_clip = all(
            float(row.get(f"brace_dual/{group}", 0.0)) >= float(lambda_max) * 0.999
            for group in ("base_solved", "boundary")
            if f"brace_dual/{group}" in row
        )
        if at_clip:
            state["lambda_clip_streak"] = int(state.get("lambda_clip_streak", 0)) + 1
        else:
            state["lambda_clip_streak"] = 0
        if state.get("lambda_clip_streak", 0) >= 100:
            return "lambda_saturated_100_steps"
    return None


def run_anchor_diagnostic(
    protocol: dict[str, Any],
    *,
    task: str,
    run_label: str,
    dataset: str,
    traced_root: Path,
    base_checkpoint: Path,
    work_dir: Path,
    job: DiagnosticJobConfig | None = None,
    brace_overrides: dict[str, Any] | None = None,
    stage: str = "anchor_feasibility",
) -> dict[str, Any]:
    from diffusion_policy.model.common.lr_scheduler import get_scheduler
    from diffusion_policy.workspace.robotworkspace import aggregate_training_loss, compute_brace_anchor_loss, module_sha256

    gate = protocol.get("constraint_feasibility", {})
    train_protocol = protocol.get("training", {})
    steps_per_epoch = int(train_protocol.get("steps_per_epoch", 247))
    probe_seed = int(gate.get("probe_seed", 42))
    holdout_fraction = float(gate.get("probe_holdout_fraction", 0.5))
    probe_num_draws = int(gate.get("probe_num_draws", 1))
    tail_steps = gate.get("tail_steps")
    tail_steps = int(tail_steps) if tail_steps is not None else None

    if job is None:
        job = DiagnosticJobConfig(
            diagnostic_steps=int(gate.get("diagnostic_steps", 200)),
            scheduler_total_steps=int(gate.get("scheduler_total_steps", gate.get("diagnostic_steps", 200))),
            checkpoint_steps=[int(step) for step in gate.get("checkpoint_steps", [])],
            checkpoint_epochs={str(k): int(v) for k, v in dict(gate.get("checkpoint_epochs", {})).items()},
            dual_lr=float(protocol.get("anchor_smoke", {}).get("dual_lr", 0.01)),
            grad_diag_at_checkpoints_only=bool(gate.get("grad_diag_at_checkpoints_only", True)),
            record_peak_memory=bool(gate.get("record_peak_memory", True)),
        )
    if brace_overrides and brace_overrides.get("dual_lr") is not None:
        job.dual_lr = float(brace_overrides["dual_lr"])
    if brace_overrides and brace_overrides.get("lambda_max") is not None:
        job.lambda_max = float(brace_overrides["lambda_max"])

    zarr_path, manifest_sha256 = ensure_screen_zarr(
        task=task,
        run_label=run_label,
        dataset=dataset,
        traced_root=traced_root,
        output_dir=work_dir,
    )
    hard_seeds_file = REPO_ROOT / "experiments/phase1/eval_results_200/hard_eval_seeds" / f"{task}.json"
    anchor_zarr_path = ensure_anchor_zarr(
        task=task,
        rollout_dir=traced_root,
        hard_seeds_file=hard_seeds_file,
        output_dir=work_dir,
    )
    anchor_manifest = json.loads((anchor_zarr_path / "brace_anchor_manifest.json").read_text(encoding="utf-8"))
    probe_split: AnchorProbeSplit = build_anchor_probe_split(
        anchor_manifest,
        split_seed=probe_seed,
        holdout_fraction=holdout_fraction,
    )
    split_path = work_dir / "anchor_probe_split.json"
    write_json_atomic(split_path, probe_split.to_dict())

    cfg = build_training_config(
        task=task,
        zarr_path=zarr_path,
        anchor_zarr_path=anchor_zarr_path,
        base_checkpoint=base_checkpoint,
        train_seed=int(protocol.get("train_seed", 0)),
        learning_rate=float(train_protocol.get("learning_rate", 5e-5)),
        batch_size=int(train_protocol.get("batch_size", 128)),
        rollout_per_batch=int(train_protocol.get("rollout_per_batch", 16)),
    )
    apply_brace_anchor_overrides(cfg, job, brace_overrides)
    cfg.training.brace_anchor.dataloader.num_batches = job.diagnostic_steps
    workspace = load_workspace(cfg)
    bootstrap_workspace(workspace, base_checkpoint)
    device = torch.device(workspace.cfg.training.device)

    if job.warmstart_target_ratio is not None and workspace.brace_dual_state is not None:
        workspace.brace_dual_state.values.zero_()

    group_map = {
        str(key): int(value)
        for key, value in dict(
            OmegaConf.select(cfg, "training.brace_anchor.groups", default={"base_solved": 1, "boundary": 2})
        ).items()
    }
    loader_cfg = OmegaConf.select(workspace.cfg, "training.brace_anchor.dataloader", default=workspace.cfg.dataloader)
    samples_per_group = int(loader_cfg.batch_size) // len(group_map)

    anchor_dataset_for_split, _ = build_anchor_dataloader(workspace, anchor_zarr_path, allowed_indices=None)
    train_indices = sequence_indices_for_env_seeds(anchor_dataset_for_split, probe_split.train_env_seeds)
    probe_indices = sequence_indices_for_env_seeds(anchor_dataset_for_split, probe_split.probe_env_seeds)
    probe_batch_idx = build_fixed_stratified_batch(
        anchor_dataset_for_split,
        probe_indices,
        group_map,
        samples_per_group,
        seed=probe_seed,
    )
    probe_batch_np = anchor_dataset_for_split[probe_batch_idx]
    probe_batch = anchor_dataset_for_split.postprocess(probe_batch_np, device)
    probe_env_seeds = sorted({int(value) for value in _anchor_env_seeds(probe_batch)})
    del anchor_dataset_for_split
    if device.type == "cuda":
        torch.cuda.empty_cache()

    probe_draws_list = []
    if job.anchor_enabled and workspace.brace_teacher is not None:
        probe_draws_list = materialize_probe_draws(
            workspace.brace_teacher,
            probe_batch,
            workspace.cfg,
            seed=probe_seed,
            device=device,
            num_draws=probe_num_draws,
        )

    sft_dataset, sft_loader = build_sft_dataloader(workspace)
    anchor_dataset, anchor_loader = (
        build_anchor_dataloader(
            workspace,
            anchor_zarr_path,
            allowed_indices=train_indices,
            seed=int(protocol.get("train_seed", 0)),
        )
        if job.anchor_enabled
        else (None, None)
    )
    sft_iter = iter_postprocessed_batches(sft_dataset, sft_loader, device)
    anchor_iter = (
        iter_postprocessed_batches(anchor_dataset, anchor_loader, device) if anchor_loader is not None else None
    )

    ema = None
    if workspace.cfg.training.use_ema:
        ema = hydra.utils.instantiate(workspace.cfg.ema, model=workspace.ema_model)
        ema.optimization_step = int(workspace.global_step)
        ema.decay = ema.get_decay(ema.optimization_step)

    grad_accum = int(workspace.cfg.training.gradient_accumulate_every)
    lr_scheduler = get_scheduler(
        workspace.cfg.training.lr_scheduler,
        optimizer=workspace.optimizer,
        num_warmup_steps=int(workspace.cfg.training.lr_warmup_steps),
        num_training_steps=job.scheduler_total_steps,
        last_epoch=int(workspace.global_step) - 1,
    )

    lambda_max = job.lambda_max
    if lambda_max is None:
        cfg_lambda_max = OmegaConf.select(workspace.cfg, "training.brace_anchor.lambda_max", default=None)
        lambda_max = float(cfg_lambda_max) if cfg_lambda_max is not None else None

    expected_group_values = set(group_map.values())
    teacher_hash_before = module_sha256(workspace.brace_teacher) if workspace.brace_teacher is not None else None
    log_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    checkpoint_artifacts: dict[str, Any] = {}
    checkpoint_paths: dict[str, str] = {}
    early_stop_state: dict[str, Any] = {}
    early_stop_reason: str | None = None
    warmstart_applied = False

    if device.type == "cuda" and job.record_peak_memory:
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(job.diagnostic_steps):
        batch = next(sft_iter)
        anchor_batch = None
        anchor_term = torch.zeros((), device=device)
        anchor_constraints: dict[str, torch.Tensor] = {}
        anchor_epsilons: dict[str, float] = {}
        anchor_monitor: dict[str, torch.Tensor] = {}

        if job.anchor_enabled and anchor_iter is not None:
            anchor_batch = next(anchor_iter)
            labels = anchor_batch["sample_preservation_group"].detach().cpu().numpy()
            present = {int(value) for value in labels}
            if present != expected_group_values:
                raise RuntimeError(
                    f"anchor batch step {step} missing preservation groups: got {sorted(present)}, "
                    f"expected {sorted(expected_group_values)}"
                )

        raw_loss, _ = aggregate_training_loss(workspace.model, batch, workspace.cfg)
        reference_student = workspace.ema_model if workspace.cfg.training.use_ema else None

        step0_grad_probe = False
        if job.anchor_enabled and anchor_batch is not None and workspace.brace_dual_state is not None:
            anchor_term, anchor_constraints, anchor_epsilons, anchor_monitor = compute_brace_anchor_loss(
                workspace.model,
                workspace.brace_teacher,
                anchor_batch,
                workspace.cfg,
                workspace.brace_dual_state,
                reference_student=reference_student,
            )
            if (
                not warmstart_applied
                and job.warmstart_target_ratio is not None
                and step == 0
            ):
                params = [param for param in workspace.model.parameters() if param.requires_grad]
                scaled_raw = raw_loss / grad_accum
                lambdas = estimate_warmstart_lambdas(
                    params,
                    scaled_raw=scaled_raw,
                    anchor_constraints=anchor_constraints,
                    grad_accum=grad_accum,
                    target_ratio=job.warmstart_target_ratio,
                )
                workspace.brace_dual_state.set_values(lambdas)
                warmstart_applied = True
                step0_grad_probe = True
            if (
                job.auto_fixed_beta_ratio is not None
                and step == 0
                and anchor_constraints
            ):
                params = [param for param in workspace.model.parameters() if param.requires_grad]
                scaled_raw = raw_loss / grad_accum
                betas = estimate_warmstart_lambdas(
                    params,
                    scaled_raw=scaled_raw,
                    anchor_constraints=anchor_constraints,
                    grad_accum=grad_accum,
                    target_ratio=job.auto_fixed_beta_ratio,
                )
                workspace.cfg.training.brace_anchor.fixed_beta = betas
                step0_grad_probe = True

            if step0_grad_probe:
                workspace.optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

        total_loss = raw_loss + anchor_term
        params = [param for param in workspace.model.parameters() if param.requires_grad]
        scaled_raw = raw_loss / grad_accum
        scaled_anchor = anchor_term / grad_accum
        scaled_total = total_loss / grad_accum

        is_checkpoint_step = bool(job.checkpoint_steps) and (step + 1) in job.checkpoint_steps
        grad_norm_sft = 0.0
        grad_norm_anchor = 0.0
        group_grad_metrics: dict[str, float | None] = {}
        grad_cosine = None

        if job.anchor_enabled and workspace.brace_dual_state is not None:
            grad_norm_sft, grad_norm_anchor = measure_basic_grad_norms(
                params,
                scaled_raw=scaled_raw,
                scaled_anchor=scaled_anchor,
            )
            if is_checkpoint_step:
                dual_values_pre = workspace.brace_dual_state.as_dict()
                group_grad_metrics = measure_checkpoint_grad_metrics(
                    params,
                    scaled_raw=scaled_raw,
                    scaled_anchor=scaled_anchor,
                    anchor_constraints=anchor_constraints,
                    grad_accum=grad_accum,
                    dual_values=dual_values_pre,
                    grad_norm_sft=grad_norm_sft,
                )
                grad_cosine = group_grad_metrics.pop("grad_cosine_sft_anchor", None)
            workspace.optimizer.zero_grad(set_to_none=True)
            scaled_total.backward()
            grad_norm_total = flatten_grad_norm(params)
        else:
            workspace.optimizer.zero_grad(set_to_none=True)
            scaled_total.backward()
            grad_norm_sft = flatten_grad_norm(params)
            grad_norm_anchor = 0.0
            grad_norm_total = grad_norm_sft

        if (step + 1) % grad_accum == 0:
            workspace.optimizer.step()
            workspace.optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
            if ema is not None:
                ema.step(workspace.model)
            if job.anchor_enabled and workspace.brace_dual_state is not None and anchor_constraints:
                workspace.brace_dual_state.update(anchor_constraints, anchor_epsilons, job.dual_lr, lambda_max=lambda_max)

        probe_raw: dict[str, float] = {}
        probe_monitor: dict[str, float] = {}
        probe_stats: dict[str, Any] = {"probe_draw_count": 0}
        if probe_draws_list:
            probe_raw, probe_monitor, probe_stats = evaluate_probe_draws(
                workspace.model,
                workspace.brace_teacher,
                probe_draws_list,
                workspace.cfg,
                reference_student=reference_student,
            )

        dual_values = workspace.brace_dual_state.as_dict() if workspace.brace_dual_state is not None else {}
        peak_mem = None
        if device.type == "cuda" and job.record_peak_memory:
            peak_mem = float(torch.cuda.max_memory_allocated(device))

        row = {
            "step": step,
            "global_step": step,
            "train_loss": float(raw_loss.detach().item()),
            "total_loss": float(total_loss.detach().item()),
            "lr": float(lr_scheduler.get_last_lr()[0]),
            "grad_norm_sft": grad_norm_sft,
            "grad_norm_anchor": grad_norm_anchor,
            "grad_norm_total": grad_norm_total,
            "grad_cosine_sft_anchor": grad_cosine,
            "anchor_to_sft_grad_ratio": grad_norm_anchor / max(grad_norm_sft, 1e-12),
            "anchor_env_seeds": _anchor_env_seeds(anchor_batch) if anchor_batch is not None else [],
            "probe_env_seeds": probe_env_seeds,
            **{k: v for k, v in group_grad_metrics.items() if v is not None},
        }
        if workspace.brace_teacher is not None:
            row["brace_teacher_sha256"] = workspace.brace_teacher_sha256
        if peak_mem is not None:
            row["cuda_peak_memory_bytes"] = peak_mem
        for group, value in anchor_constraints.items():
            row[f"brace_constraint/{group}"] = float(value.detach().item())
        for key, value in anchor_monitor.items():
            row[f"brace_monitor/{key}"] = float(value.detach().item())
        for group, value in dual_values.items():
            row[f"brace_dual/{group}"] = float(value)
        for group, value in probe_raw.items():
            row[f"probe_constraint/{group}"] = float(value)
        for key, value in probe_monitor.items():
            row[f"probe_monitor/{key}"] = float(value)
        for key, value in probe_stats.items():
            if key != "per_draw":
                row[key] = value
        log_rows.append(row)

        probe_row = {
            "step": step,
            **{f"brace_constraint/{group}": value for group, value in probe_raw.items()},
            **{f"brace_monitor/{key}": value for key, value in probe_monitor.items()},
        }
        for key, value in probe_stats.items():
            if key.startswith("probe_") and key != "per_draw":
                probe_row[key] = value
        probe_rows.append(probe_row)

        early_stop_reason = _check_early_stop(row=row, state=early_stop_state, job=job, lambda_max=lambda_max)
        if early_stop_reason:
            write_json_atomic(
                work_dir / "early_stop.json",
                {"step": step + 1, "reason": early_stop_reason, "row": row},
            )
            break

        if is_checkpoint_step:
            ckpt_dir = work_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            workspace.global_step = step + 1
            workspace.epoch = _epoch_for_step(step + 1, job.checkpoint_epochs, steps_per_epoch)
            ckpt_path = ckpt_dir / f"step_{step + 1}.ckpt"
            workspace.save_checkpoint(path=ckpt_path, use_thread=False)
            checkpoint_paths[str(step + 1)] = str(ckpt_path)
            checkpoint_artifacts[str(step + 1)] = {
                "epoch": workspace.epoch,
                "probe_constraints": dict(probe_raw),
                "probe_monitor": dict(probe_monitor),
                "probe_stats": {k: v for k, v in probe_stats.items() if k != "per_draw"},
                "dual": dict(dual_values),
                "lr": float(lr_scheduler.get_last_lr()[0]),
                "checkpoint_path": str(ckpt_path),
                "anchor_to_sft_grad_ratio": row["anchor_to_sft_grad_ratio"],
            }

    teacher_hash_after = module_sha256(workspace.brace_teacher) if workspace.brace_teacher is not None else None
    teacher_hash_stable = (
        teacher_hash_before == teacher_hash_after == workspace.brace_teacher_sha256
        if teacher_hash_before is not None
        else True
    )
    epsilon = float(protocol.get("anchor_smoke", {}).get("identity_epsilon", 1e-4))

    trajectory_path = work_dir / "feasibility_trajectory.jsonl"
    with trajectory_path.open("w", encoding="utf-8") as handle:
        for row in log_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    probe_trajectory_path = work_dir / "feasibility_probe_trajectory.jsonl"
    with probe_trajectory_path.open("w", encoding="utf-8") as handle:
        for row in probe_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    feasibility_train = evaluate_constraint_feasibility(log_rows, protocol)
    feasibility_probe = evaluate_constraint_feasibility(probe_rows, protocol)
    trajectory_summary = summarize_feasibility_trajectory(
        log_rows, epsilon=epsilon, tail_steps=tail_steps, prefix="train_"
    )
    probe_summary = summarize_feasibility_trajectory(
        probe_rows, epsilon=epsilon, tail_steps=tail_steps, prefix="probe_"
    )

    passed = bool(feasibility_train["passed"] and teacher_hash_stable)
    return {
        "schema_version": 3,
        "stage": stage,
        "job_id": job.job_id,
        "exploratory": bool(protocol.get("exploratory", False)),
        "promotion_eligible": bool(protocol.get("promotion_eligible", False)),
        "protocol_revision": protocol.get("protocol_revision"),
        "passed": passed,
        "complete": early_stop_reason is None,
        "early_stop_reason": early_stop_reason,
        "diagnostic_steps": job.diagnostic_steps,
        "steps_completed": len(log_rows),
        "scheduler_total_steps": job.scheduler_total_steps,
        "feasibility": feasibility_train,
        "feasibility_on_probe": feasibility_probe,
        "trajectory_summary": trajectory_summary,
        "probe_summary": probe_summary,
        "gate_note": (
            "screen.v1.2 still gates on full-trajectory train-batch constraints using identity_epsilon; "
            "held-out probe tail summaries are forensic-only until a calibrated protocol is frozen."
        ),
        "dataset_manifest_sha256": manifest_sha256,
        "anchor_manifest_sha256": anchor_manifest.get("manifest_sha256"),
        "probe_split_sha256": probe_split.split_sha256,
        "probe_split_path": str(split_path),
        "rollout_dir": str(traced_root),
        "teacher_hash_before": teacher_hash_before,
        "teacher_hash_end": teacher_hash_after,
        "teacher_hash_stable": teacher_hash_stable,
        "probe_seed": probe_seed,
        "probe_num_draws": probe_num_draws,
        "probe_env_seeds": probe_env_seeds,
        "train_env_seed_count": len(probe_split.train_env_seeds),
        "probe_env_seed_count": len(probe_split.probe_env_seeds),
        "checkpoint_steps": job.checkpoint_steps,
        "checkpoint_artifacts": checkpoint_artifacts,
        "checkpoint_paths": checkpoint_paths,
        "trajectory_path": str(trajectory_path),
        "probe_trajectory_path": str(probe_trajectory_path),
        "git_commit": git_commit(),
    }
