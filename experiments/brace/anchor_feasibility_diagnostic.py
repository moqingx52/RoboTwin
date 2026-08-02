"""Training-path faithful anchor feasibility diagnostic with full trajectory logging."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]

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


def _flatten_grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is not None:
            total += float(param.grad.detach().data.norm(2).item() ** 2)
    return math.sqrt(total)


def _cosine_similarity(left: torch.Tensor | None, right: torch.Tensor | None) -> float | None:
    if left is None or right is None or left.numel() == 0 or right.numel() == 0:
        return None
    denom = float(left.norm() * right.norm())
    if denom <= 0:
        return None
    return float(torch.dot(left, right).item() / denom)


def _anchor_env_seeds(batch: dict[str, Any]) -> list[int]:
    seeds = batch.get("sample_env_seed")
    if seeds is None:
        return []
    return [int(value) for value in seeds.detach().cpu().tolist()]


def _grad_vector_from_grads(grads) -> torch.Tensor | None:
    chunks = [grad.detach().reshape(-1) for grad in grads if grad is not None]
    if not chunks:
        return None
    return torch.cat(chunks, dim=0)


def run_feasibility_diagnostic(
    protocol: dict[str, Any],
    *,
    task: str,
    run_label: str,
    dataset: str,
    traced_root: Path,
    base_checkpoint: Path,
    work_dir: Path,
) -> dict[str, Any]:
    from diffusion_policy.model.common.lr_scheduler import get_scheduler
    from diffusion_policy.workspace.robotworkspace import aggregate_training_loss, compute_brace_anchor_loss, module_sha256

    gate = protocol.get("constraint_feasibility", {})
    steps = int(gate.get("diagnostic_steps", 200))
    scheduler_total_steps = int(gate.get("scheduler_total_steps", steps))
    probe_seed = int(gate.get("probe_seed", 42))
    holdout_fraction = float(gate.get("probe_holdout_fraction", 0.5))
    checkpoint_steps = [int(step) for step in gate.get("checkpoint_steps", [])]
    train_protocol = protocol.get("training", {})
    dual_lr = float(protocol["anchor_smoke"]["dual_lr"])

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
    cfg.training.brace_anchor.dataloader.num_batches = steps
    workspace = load_workspace(cfg)
    bootstrap_workspace(workspace, base_checkpoint)
    device = torch.device(workspace.cfg.training.device)

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
    probe_draws = materialize_probe_draws(
        workspace.brace_teacher,
        probe_batch,
        workspace.cfg,
        seed=probe_seed,
        device=device,
    )

    sft_dataset, sft_loader = build_sft_dataloader(workspace)
    anchor_dataset, anchor_loader = build_anchor_dataloader(
        workspace,
        anchor_zarr_path,
        allowed_indices=train_indices,
        seed=int(protocol.get("train_seed", 0)),
    )
    sft_iter = iter_postprocessed_batches(sft_dataset, sft_loader, device)
    anchor_iter = iter_postprocessed_batches(anchor_dataset, anchor_loader, device)

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
        num_training_steps=scheduler_total_steps,
        last_epoch=int(workspace.global_step) - 1,
    )

    expected_group_values = set(group_map.values())
    teacher_hash_before = module_sha256(workspace.brace_teacher)
    log_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    checkpoint_artifacts: dict[str, Any] = {}

    for step in range(steps):
        batch = next(sft_iter)
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
        anchor_term, anchor_constraints, anchor_epsilons, anchor_monitor = compute_brace_anchor_loss(
            workspace.model,
            workspace.brace_teacher,
            anchor_batch,
            workspace.cfg,
            workspace.brace_dual_state,
            reference_student=reference_student,
        )
        total_loss = raw_loss + anchor_term
        params = [param for param in workspace.model.parameters() if param.requires_grad]
        scaled_raw = raw_loss / grad_accum
        scaled_anchor = anchor_term / grad_accum
        scaled_total = total_loss / grad_accum

        sft_grads = torch.autograd.grad(scaled_raw, params, retain_graph=True, allow_unused=True)
        anchor_grads = torch.autograd.grad(scaled_anchor, params, retain_graph=True, allow_unused=True)
        sft_grad_vec = _grad_vector_from_grads(sft_grads)
        anchor_grad_vec = _grad_vector_from_grads(anchor_grads)
        grad_norm_sft = float(sft_grad_vec.norm()) if sft_grad_vec is not None else 0.0
        grad_norm_anchor = float(anchor_grad_vec.norm()) if anchor_grad_vec is not None else 0.0
        grad_cosine = _cosine_similarity(sft_grad_vec, anchor_grad_vec)

        group_grad_metrics: dict[str, float] = {}
        for group, constraint in anchor_constraints.items():
            group_grads = torch.autograd.grad(
                constraint / grad_accum,
                params,
                retain_graph=True,
                allow_unused=True,
            )
            group_vec = _grad_vector_from_grads(group_grads)
            group_norm = float(group_vec.norm()) if group_vec is not None else 0.0
            group_grad_metrics[f"grad_norm_constraint/{group}"] = group_norm
            group_grad_metrics[f"grad_cosine_sft_{group}"] = _cosine_similarity(sft_grad_vec, group_vec)
            dual_value = float(workspace.brace_dual_state.as_dict().get(group, 0.0))
            group_grad_metrics[f"lambda_grad_scale/{group}"] = (
                dual_value * group_norm / max(grad_norm_sft, 1e-12)
            )

        workspace.optimizer.zero_grad(set_to_none=True)
        scaled_total.backward()
        grad_norm_total = _flatten_grad_norm(params)
        if (step + 1) % grad_accum == 0:
            workspace.optimizer.step()
            workspace.optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
            if ema is not None:
                ema.step(workspace.model)
            workspace.brace_dual_state.update(anchor_constraints, anchor_epsilons, dual_lr)

        probe_raw, probe_monitor = evaluate_probe_draws(
            workspace.model,
            workspace.brace_teacher,
            probe_draws,
            workspace.cfg,
            reference_student=reference_student,
        )
        dual_values = workspace.brace_dual_state.as_dict()
        row = {
            "step": step,
            "global_step": step,
            "train_loss": float(raw_loss.detach().item()),
            "total_loss": float(total_loss.detach().item()),
            "lr": float(lr_scheduler.get_last_lr()[0]),
            "brace_teacher_sha256": workspace.brace_teacher_sha256,
            "grad_norm_sft": grad_norm_sft,
            "grad_norm_anchor": grad_norm_anchor,
            "grad_norm_total": grad_norm_total,
            "grad_cosine_sft_anchor": grad_cosine,
            "anchor_to_sft_grad_ratio": grad_norm_anchor / max(grad_norm_sft, 1e-12),
            "anchor_env_seeds": _anchor_env_seeds(anchor_batch),
            "probe_env_seeds": probe_env_seeds,
            **group_grad_metrics,
        }
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
        log_rows.append(row)
        probe_rows.append(
            {
                "step": step,
                **{f"brace_constraint/{group}": value for group, value in probe_raw.items()},
                **{f"brace_monitor/{key}": value for key, value in probe_monitor.items()},
            }
        )
        if checkpoint_steps and (step + 1) in checkpoint_steps:
            checkpoint_artifacts[str(step + 1)] = {
                "probe_constraints": dict(probe_raw),
                "dual": dict(dual_values),
                "lr": float(lr_scheduler.get_last_lr()[0]),
            }

    teacher_hash_after = module_sha256(workspace.brace_teacher)
    teacher_hash_stable = teacher_hash_before == teacher_hash_after == workspace.brace_teacher_sha256
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
    trajectory_summary = summarize_feasibility_trajectory(log_rows, epsilon=epsilon, prefix="train_")
    probe_summary = summarize_feasibility_trajectory(probe_rows, epsilon=epsilon, prefix="probe_")

    passed = bool(feasibility_train["passed"] and teacher_hash_stable)
    return {
        "schema_version": 3,
        "stage": "anchor_feasibility",
        "exploratory": bool(protocol.get("exploratory", False)),
        "protocol_revision": protocol.get("protocol_revision"),
        "passed": passed,
        "complete": True,
        "diagnostic_steps": steps,
        "scheduler_total_steps": scheduler_total_steps,
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
        "probe_env_seeds": probe_env_seeds,
        "train_env_seed_count": len(probe_split.train_env_seeds),
        "probe_env_seed_count": len(probe_split.probe_env_seeds),
        "checkpoint_steps": checkpoint_steps,
        "checkpoint_artifacts": checkpoint_artifacts,
        "trajectory_path": str(trajectory_path),
        "probe_trajectory_path": str(probe_trajectory_path),
        "git_commit": git_commit(),
    }
