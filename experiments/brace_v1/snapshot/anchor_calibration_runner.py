#!/usr/bin/env python3
"""Run a single BRACE anchor calibration diagnostic job (Phase 3A)."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.anchor_diagnostic_loop import DiagnosticJobConfig, run_anchor_diagnostic
from experiments.brace.build_screen_dataset import sha256 as file_sha256
from experiments.brace.replay_audit import git_commit, read_json, repo_path, write_json_atomic


def job_config_from_manifest(job: dict[str, Any], protocol: dict[str, Any]) -> DiagnosticJobConfig:
    gate = protocol.get("constraint_feasibility", {})
    brace = job.get("brace_overrides", {})
    dual_lr = float(brace.get("dual_lr", protocol.get("anchor_smoke", {}).get("dual_lr", 0.01)))
    return DiagnosticJobConfig(
        job_id=str(job["job_id"]),
        anchor_enabled=bool(job.get("anchor_enabled", True)),
        diagnostic_steps=int(job.get("diagnostic_steps", gate.get("diagnostic_steps", 1235))),
        scheduler_total_steps=int(gate.get("scheduler_total_steps", 2470)),
        checkpoint_steps=[int(step) for step in job.get("checkpoint_steps", gate.get("checkpoint_steps", []))],
        checkpoint_epochs={str(k): int(v) for k, v in dict(gate.get("checkpoint_epochs", {})).items()},
        dual_lr=dual_lr,
        lambda_max=job.get("lambda_max", brace.get("lambda_max")),
        warmstart_target_ratio=job.get("warmstart_target_ratio"),
        auto_fixed_beta_ratio=job.get("auto_fixed_beta_ratio"),
        grad_diag_at_checkpoints_only=bool(gate.get("grad_diag_at_checkpoints_only", True)),
        record_peak_memory=bool(gate.get("record_peak_memory", True)),
        early_stop_enabled=bool(job.get("early_stop_enabled", True)),
    )


def run_calibration_job(
    protocol: dict[str, Any],
    job: dict[str, Any],
    *,
    task: str,
    run_label: str,
    dataset: str,
    traced_root: Path,
    base_checkpoint: Path,
    work_dir: Path,
    training_seed: int | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "schema_version": 3,
            "stage": "anchor_calibration",
            "job_id": job.get("job_id"),
            "passed": False,
            "complete": False,
            "error": "CUDA is required for anchor calibration",
            "git_commit": git_commit(),
        }

    job_cfg = job_config_from_manifest(job, protocol)
    brace_overrides = dict(job.get("brace_overrides", {}))
    resolved_path = work_dir / "resolved_config.json"
    write_json_atomic(
        resolved_path,
        {
            "job": job,
            "job_config": job_cfg.__dict__,
            "brace_overrides": brace_overrides,
        },
    )
    summary = run_anchor_diagnostic(
        protocol,
        task=task,
        run_label=run_label,
        dataset=dataset,
        traced_root=traced_root,
        base_checkpoint=base_checkpoint,
        work_dir=work_dir,
        job=job_cfg,
        brace_overrides=brace_overrides,
        train_seed=training_seed,
        stage="anchor_calibration",
    )
    summary["training_seed"] = training_seed if training_seed is not None else int(protocol.get("train_seed", 0))
    summary["resolved_config_path"] = str(resolved_path)
    summary["config_sha256"] = file_sha256(resolved_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=BRACE_DIR / "screen_protocol.v1.3.exploratory_calibration.json")
    parser.add_argument("--jobs", type=Path, default=BRACE_DIR / "calibration_jobs.place_container_plate.v1.json")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--task", default="place_container_plate")
    parser.add_argument("--run-label", default="place_pilot_v2.3")
    parser.add_argument("--dataset", default="N1")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--traced-rollout-dir", type=Path, default=BRACE_DIR / "rollouts_traced")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()

    protocol = read_json(repo_path(args.protocol))
    manifest = read_json(repo_path(args.jobs))
    manifest_protocol = manifest.get("protocol_path")
    if manifest_protocol:
        manifest_protocol_path = repo_path(Path(str(manifest_protocol)))
        if not manifest_protocol_path.is_file() or file_sha256(manifest_protocol_path) != file_sha256(repo_path(args.protocol)):
            raise SystemExit("jobs manifest protocol_path does not match --protocol")
    if manifest.get("training_seeds") is not None:
        manifest_seeds = [int(seed) for seed in manifest["training_seeds"]]
        protocol_seeds = [int(seed) for seed in protocol.get("training_seeds", [])]
        if manifest_seeds != protocol_seeds:
            raise SystemExit("jobs manifest training_seeds do not match protocol")
    job = next((row for row in manifest["jobs"] if row["job_id"] == args.job_id), None)
    if job is None:
        raise SystemExit(f"unknown calibration job_id: {args.job_id}")

    output = repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    work_dir = repo_path(args.work_dir) if args.work_dir else output.parent

    summary = run_calibration_job(
        protocol,
        job,
        task=args.task,
        run_label=args.run_label,
        dataset=args.dataset,
        traced_root=repo_path(args.traced_rollout_dir),
        base_checkpoint=repo_path(args.checkpoint),
        work_dir=work_dir,
        training_seed=int(job["training_seed"]) if "training_seed" in job else None,
    )
    summary["protocol_path"] = str(repo_path(args.protocol))
    summary["protocol_sha256"] = file_sha256(repo_path(args.protocol))
    summary["jobs_manifest_path"] = str(repo_path(args.jobs))
    summary["jobs_manifest_sha256"] = file_sha256(repo_path(args.jobs))
    write_json_atomic(output, summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("complete", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
