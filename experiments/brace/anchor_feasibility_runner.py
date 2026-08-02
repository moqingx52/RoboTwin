#!/usr/bin/env python3
"""Short GPU diagnostic that runs pre-registered anchor feasibility optimizer steps."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
DP_DIR = REPO_ROOT / "policy" / "DP"
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DP_DIR) not in sys.path:
    sys.path.insert(0, str(DP_DIR))

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
from experiments.brace.build_screen_dataset import sha256 as file_sha256
from experiments.brace.replay_audit import git_commit, read_json, repo_path, write_json_atomic
from experiments.brace.screen_gates import evaluate_constraint_feasibility


def run_anchor_feasibility(
    protocol: dict,
    *,
    task: str,
    run_label: str,
    dataset: str,
    traced_root: Path,
    base_checkpoint: Path,
    work_dir: Path | None = None,
) -> dict:
    if not torch.cuda.is_available():
        return {
            "schema_version": 1,
            "stage": "anchor_feasibility",
            "passed": False,
            "complete": False,
            "error": "CUDA is required for anchor feasibility diagnostic",
            "git_commit": git_commit(),
        }

    gate = protocol.get("constraint_feasibility", {})
    steps = int(gate.get("diagnostic_steps", 200))
    train_protocol = protocol.get("training", {})
    cleanup = None
    if work_dir is None:
        cleanup = tempfile.TemporaryDirectory()
        work_dir = Path(cleanup.name)
    work_dir.mkdir(parents=True, exist_ok=True)

    from diffusion_policy.workspace.robotworkspace import aggregate_training_loss, compute_brace_anchor_loss, module_sha256

    try:
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
        anchor_manifest = json.loads(
            (anchor_zarr_path / "brace_anchor_manifest.json").read_text(encoding="utf-8")
        )
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
        sft_dataset, sft_loader = build_sft_dataloader(workspace)
        anchor_dataset, anchor_loader = build_anchor_dataloader(workspace, anchor_zarr_path)
        sft_iter = iter_postprocessed_batches(sft_dataset, sft_loader, device)
        anchor_iter = iter_postprocessed_batches(anchor_dataset, anchor_loader, device)
        ema = None
        if workspace.cfg.training.use_ema:
            ema = hydra.utils.instantiate(workspace.cfg.ema, model=workspace.ema_model)
            ema.optimization_step = int(workspace.global_step)
            ema.decay = ema.get_decay(ema.optimization_step)

        group_ids = {
            str(key): int(value)
            for key, value in dict(
                OmegaConf.select(cfg, "training.brace_anchor.groups", default={"base_solved": 1, "boundary": 2})
            ).items()
        }
        expected_group_values = set(group_ids.values())
        dual_lr = float(protocol["anchor_smoke"]["dual_lr"])
        teacher_hash_before = module_sha256(workspace.brace_teacher)
        log_rows = []
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
            workspace.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            workspace.optimizer.step()
            workspace.optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.step(workspace.model)
            workspace.brace_dual_state.update(anchor_constraints, anchor_epsilons, dual_lr)
            row = {
                "step": step,
                "brace_teacher_sha256": workspace.brace_teacher_sha256,
            }
            for group, value in anchor_constraints.items():
                row[f"brace_constraint/{group}"] = float(value.detach().item())
            for key, value in anchor_monitor.items():
                row[f"brace_monitor/{key}"] = float(value.detach().item())
            log_rows.append(row)

        teacher_hash_after = module_sha256(workspace.brace_teacher)
        teacher_hash_stable = (
            teacher_hash_before == teacher_hash_after == workspace.brace_teacher_sha256
        )
        feasibility = evaluate_constraint_feasibility(log_rows, protocol)
        passed = bool(feasibility["passed"] and teacher_hash_stable)
        return {
            "schema_version": 1,
            "stage": "anchor_feasibility",
            "protocol_revision": protocol.get("protocol_revision"),
            "passed": passed,
            "complete": True,
            "diagnostic_steps": steps,
            "feasibility": feasibility,
            "dataset_manifest_sha256": manifest_sha256,
            "anchor_manifest_sha256": anchor_manifest.get("manifest_sha256"),
            "teacher_hash_before": teacher_hash_before,
            "teacher_hash_end": teacher_hash_after,
            "teacher_hash_stable": teacher_hash_stable,
            "git_commit": git_commit(),
        }
    finally:
        if cleanup is not None:
            cleanup.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("experiments/brace/screen_protocol.v1.2.json"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--dataset", default="N1")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--traced-rollout-dir", type=Path, default=BRACE_DIR / "rollouts_traced_pilot")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = repo_path(args.protocol)
    protocol = read_json(protocol_path)
    summary = run_anchor_feasibility(
        protocol,
        task=args.task,
        run_label=args.run_label,
        dataset=args.dataset,
        traced_root=repo_path(args.traced_rollout_dir),
        base_checkpoint=repo_path(args.checkpoint),
    )
    summary["protocol_path"] = str(protocol_path)
    summary["protocol_sha256"] = file_sha256(protocol_path)
    output = repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
