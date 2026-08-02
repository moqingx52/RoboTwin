#!/usr/bin/env python3
"""GPU training-path anchor smoke using real DP checkpoints and screen batches."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import dill
import torch
import yaml
from hydra import compose, initialize
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
DP_DIR = REPO_ROOT / "policy" / "DP"
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DP_DIR) not in sys.path:
    sys.path.insert(0, str(DP_DIR))

from experiments.brace.build_screen_dataset import build_dataset, sha256 as file_sha256
from experiments.brace.replay_audit import git_commit, read_json, repo_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_camera_config(camera_type: str) -> dict[str, Any]:
    camera_config_path = REPO_ROOT / "task_config" / "_camera_config.yml"
    with camera_config_path.open(encoding="utf-8") as handle:
        args = yaml.load(handle.read(), Loader=yaml.FullLoader)
    return args[camera_type]


def build_training_config(
    *,
    task: str,
    zarr_path: Path,
    base_checkpoint: Path,
    train_seed: int,
    learning_rate: float,
    batch_size: int,
    rollout_per_batch: int,
) -> OmegaConf:
    previous = os.getcwd()
    os.chdir(DP_DIR)
    try:
        with initialize(version_base=None, config_path="diffusion_policy/config"):
            cfg = compose(
                config_name="robot_dp_14",
                overrides=[
                    f"task.name={task}",
                    f"task.dataset.zarr_path={zarr_path}",
                    "task.dataset.load_to_memory=False",
                    f"dataloader.batch_size={batch_size}",
                    "dataloader.num_batches=1",
                    f"dataloader.rollout_per_batch={rollout_per_batch}",
                    "training.debug=False",
                    f"training.seed={train_seed}",
                    "training.device=cuda:0",
                    "training.resume=False",
                    f"training.resume_from_ckpt={base_checkpoint}",
                    "training.resume_training_ckpt=null",
                    "training.checkpoint_name=anchor_smoke",
                    "training.num_epochs=1",
                    "training.stop_after_epoch=1",
                    "training.checkpoint_every=1",
                    "training.normalizer_source=checkpoint",
                    "training.loss_mode=pooled",
                    "training.brace_anchor.enabled=true",
                    f"optimizer.lr={learning_rate}",
                    "exp_name=anchor_smoke",
                    "logging.mode=offline",
                    "setting=demo_clean",
                    "expert_data_num=200",
                    "head_camera_type=D435",
                ],
            )
        head_camera_cfg = get_camera_config(cfg.head_camera_type)
        cfg.task.image_shape = [3, head_camera_cfg["h"], head_camera_cfg["w"]]
        cfg.task.shape_meta.obs.head_cam.shape = [3, head_camera_cfg["h"], head_camera_cfg["w"]]
        OmegaConf.resolve(cfg)
        return cfg
    finally:
        os.chdir(previous)


def ensure_screen_zarr(
    *,
    task: str,
    run_label: str,
    dataset: str,
    traced_root: Path,
    output_dir: Path,
) -> tuple[Path, str]:
    expert = REPO_ROOT / "policy" / "DP" / "data_phase1_200" / f"{task}-expert_only.zarr"
    manifest = BRACE_DIR / "datasets" / f"{run_label}_{dataset}.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing screen manifest: {manifest}")
    zarr_path = output_dir / f"{task}_{dataset}.zarr"
    if not (zarr_path / "brace_dataset_manifest.json").is_file():
        build_dataset(
            expert,
            manifest,
            zarr_path,
            traced_root=traced_root,
        )
    return zarr_path, file_sha256(manifest)


def load_workspace(cfg: OmegaConf):
    from diffusion_policy.workspace.robotworkspace import RobotWorkspace

    workspace = RobotWorkspace(cfg)
    return workspace


def bootstrap_workspace(workspace, base_checkpoint: Path) -> None:
    workspace.load_checkpoint(
        path=base_checkpoint,
        exclude_keys=("optimizer",),
        include_keys=(),
    )
    workspace.global_step = 0
    workspace.epoch = 0
    workspace.teacher_checkpoint_path = str(base_checkpoint.resolve())
    workspace.teacher_reference = "raw"
    workspace.brace_teacher.load_state_dict(workspace.model.state_dict())
    from diffusion_policy.workspace.robotworkspace import module_sha256

    workspace.brace_teacher_sha256 = module_sha256(workspace.brace_teacher)
    workspace.brace_dual_state.values.zero_()
    device = torch.device(workspace.cfg.training.device)
    workspace.model.to(device)
    workspace.brace_teacher.to(device)
    workspace.brace_teacher.eval()
    workspace.brace_teacher.requires_grad_(False)
    if workspace.ema_model is not None:
        workspace.ema_model.to(device)


def fetch_batch(workspace):
    from diffusion_policy.dataset.base_dataset import BaseImageDataset
    from diffusion_policy.workspace.robotworkspace import apply_normalizer_from_config, create_dataloader
    import hydra

    dataset = hydra.utils.instantiate(workspace.cfg.task.dataset)
    assert isinstance(dataset, BaseImageDataset)
    apply_normalizer_from_config(
        workspace.model,
        workspace.ema_model if workspace.cfg.training.use_ema else None,
        dataset,
        workspace.cfg,
    )
    train_dataloader = create_dataloader(dataset, **workspace.cfg.dataloader)
    batch = next(iter(train_dataloader))
    device = torch.device(workspace.cfg.training.device)
    return dataset.postprocess(batch, device)


def run_training_path_smoke(
    protocol: dict[str, Any],
    *,
    task: str,
    run_label: str,
    dataset: str,
    traced_root: Path,
    base_checkpoint: Path,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "schema_version": 1,
            "smoke_level": "training_path",
            "eligible_for_screen_gate": True,
            "protocol_revision": protocol.get("protocol_revision", "screen.v1"),
            "passed": False,
            "complete": False,
            "error": "CUDA is required for training-path anchor smoke",
            "git_commit": git_commit(),
        }

    smoke_cfg = protocol["anchor_smoke"]
    training_cfg = smoke_cfg.get("training_path", {})
    train_protocol = protocol.get("training", {})
    violation_steps = int(training_cfg.get("violation_steps", smoke_cfg.get("violation_steps", 10)))
    optimizer_steps = int(training_cfg.get("optimizer_steps", 1))
    step_count = max(optimizer_steps, violation_steps)
    epsilon = float(smoke_cfg["identity_epsilon"])
    dual_lr = float(smoke_cfg["dual_lr"])

    from diffusion_policy.workspace.robotworkspace import (
        aggregate_training_loss,
        compute_brace_anchor_loss,
        module_sha256,
    )

    cleanup_dir = None
    if work_dir is None:
        cleanup_dir = tempfile.TemporaryDirectory()
        work_dir = Path(cleanup_dir.name)
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        zarr_path, manifest_sha256 = ensure_screen_zarr(
            task=task,
            run_label=run_label,
            dataset=dataset,
            traced_root=traced_root,
            output_dir=work_dir,
        )
        cfg = build_training_config(
            task=task,
            zarr_path=zarr_path,
            base_checkpoint=base_checkpoint,
            train_seed=int(protocol.get("train_seed", 0)),
            learning_rate=float(train_protocol.get("learning_rate", 5e-5)),
            batch_size=int(train_protocol.get("batch_size", 128)),
            rollout_per_batch=int(train_protocol.get("rollout_per_batch", 16)),
        )
        workspace = load_workspace(cfg)
        bootstrap_workspace(workspace, base_checkpoint)
        batch = fetch_batch(workspace)
        teacher_hash_before = module_sha256(workspace.brace_teacher)

        anchor_term, identity_constraints, _ = compute_brace_anchor_loss(
            workspace.model,
            workspace.brace_teacher,
            batch,
            workspace.cfg,
            workspace.brace_dual_state,
        )
        identity_violation = max(float(value.detach()) for value in identity_constraints.values())
        identity_passed = identity_violation <= epsilon

        with torch.no_grad():
            for parameter in workspace.model.parameters():
                if parameter.requires_grad:
                    parameter.add_(1e-3)
                    break

        dual_history: list[float] = []
        violation_history: list[float] = []
        for _ in range(step_count):
            raw_loss, _ = aggregate_training_loss(workspace.model, batch, workspace.cfg)
            anchor_term, anchor_constraints, anchor_epsilons = compute_brace_anchor_loss(
                workspace.model,
                workspace.brace_teacher,
                batch,
                workspace.cfg,
                workspace.brace_dual_state,
            )
            total_loss = raw_loss + anchor_term
            workspace.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            workspace.optimizer.step()
            workspace.optimizer.zero_grad(set_to_none=True)
            workspace.brace_dual_state.update(anchor_constraints, anchor_epsilons, dual_lr)
            violation_history.append(
                max(float(value.detach()) - anchor_epsilons[group] for group, value in anchor_constraints.items())
            )
            dual_history.append(max(workspace.brace_dual_state.as_dict().values()))

        teacher_hash_after = module_sha256(workspace.brace_teacher)
        teacher_no_grad = all(not parameter.requires_grad for parameter in workspace.brace_teacher.parameters())

        ckpt_path = work_dir / "resume.ckpt"
        workspace.epoch = 1
        workspace.global_step = 7
        workspace.save_checkpoint(path=ckpt_path, use_thread=False)

        resumed = load_workspace(cfg)
        resumed.load_checkpoint(path=ckpt_path)
        resumed_hash = module_sha256(resumed.brace_teacher)
        dual_match = torch.allclose(
            resumed.brace_dual_state.values,
            workspace.brace_dual_state.values.cpu(),
        )
        state_match = (
            resumed.global_step == workspace.global_step
            and resumed.epoch == workspace.epoch
            and resumed_hash == workspace.brace_teacher_sha256
        )

        checks = {
            "identity_constraint_near_zero": identity_passed,
            "teacher_no_grad_enforced": teacher_no_grad,
            "shared_noise_timestep_configured": identity_passed,
            "dual_nonnegativity": all(value >= -1e-12 for value in dual_history),
            "dual_rises_on_violation": bool(dual_history) and dual_history[-1] >= dual_history[0],
            "teacher_hash_stable_on_resume": teacher_hash_before == teacher_hash_after == resumed_hash,
            "checkpoint_round_trip": state_match and dual_match,
        }
        passed = all(checks.values())
        return {
            "schema_version": 1,
            "smoke_level": "training_path",
            "eligible_for_screen_gate": True,
            "protocol_revision": protocol.get("protocol_revision", "screen.v1"),
            "passed": passed,
            "complete": True,
            "checks": checks,
            "identity_violation": identity_violation,
            "dual_final": dual_history[-1] if dual_history else 0.0,
            "base_checkpoint_sha256": sha256_file(base_checkpoint),
            "dataset_manifest_sha256": manifest_sha256,
            "teacher_reference": "raw",
            "task": task,
            "run_label": run_label,
            "dataset": dataset,
            "git_commit": git_commit(),
        }
    finally:
        if cleanup_dir is not None:
            cleanup_dir.cleanup()


def main() -> int:
    import argparse
    import json

    from experiments.brace.replay_audit import write_json_atomic

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("experiments/brace/screen_protocol.v1.1.json"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--dataset", default="N1")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--traced-rollout-dir", type=Path, default=BRACE_DIR / "rollouts_traced_pilot")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = read_json(repo_path(args.protocol))
    summary = run_training_path_smoke(
        protocol,
        task=args.task,
        run_label=args.run_label,
        dataset=args.dataset,
        traced_root=repo_path(args.traced_rollout_dir),
        base_checkpoint=repo_path(args.checkpoint),
        work_dir=repo_path(args.work_dir) if args.work_dir else None,
    )
    output = repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
