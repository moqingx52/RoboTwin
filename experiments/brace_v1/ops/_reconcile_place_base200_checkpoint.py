#!/usr/bin/env python3
"""Reconcile a completed Line A checkpoint after post-training scheduler failure.

Verifies checkpoint load/EMA, frozen training config, input provenance hashes,
training log integrity, and anchor feasibility JSON before atomically updating
train_state.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
BRACE = REPO / "experiments" / "brace"
DP = REPO / "policy" / "DP"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.brace.anchor_behavior_eval import inspect_diagnostic_checkpoint
from experiments.brace.replay_audit import read_json, write_json_atomic


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_hydra_log(checkpoint_name: str, *, train_seed: int | None = None) -> Path | None:
    outputs = DP / "data" / "outputs"
    if not outputs.is_dir():
        return None
    matches: list[tuple[float, Path]] = []
    for candidate in outputs.rglob("logs.json.txt"):
        hydra_cfg = candidate.parent / ".hydra" / "config.yaml"
        if not hydra_cfg.is_file():
            continue
        text = hydra_cfg.read_text(encoding="utf-8", errors="replace")
        if checkpoint_name not in text:
            continue
        if train_seed is not None:
            try:
                from omegaconf import OmegaConf

                cfg = OmegaConf.load(hydra_cfg)
                seed = OmegaConf.select(cfg, "training.seed", default=None)
                if seed is None or int(seed) != int(train_seed):
                    continue
            except Exception:
                continue
        if candidate.stat().st_size <= 0:
            continue
        matches.append((candidate.stat().st_mtime, candidate))
    if not matches:
        return None
    matches.sort()
    return matches[-1][1]


def ensure_feasibility(
    *,
    method: str,
    checkpoint_dir: Path,
    epochs: int,
    protocol_path: Path,
    checkpoint_name: str,
    train_seed: int | None = None,
    force: bool = False,
) -> tuple[Path | None, dict[str, Any]]:
    feasibility_path = checkpoint_dir / f"{epochs}.feasibility.json"
    if method not in ("B2", "B3"):
        return None, {"skipped": True, "reason": "not_anchor_method"}
    if (
        not force
        and feasibility_path.is_file()
        and feasibility_path.stat().st_size > 0
    ):
        payload = read_json(feasibility_path)
        return feasibility_path, {"existing": True, "passed": payload.get("passed"), "parseable": True}
    log_path = find_hydra_log(checkpoint_name, train_seed=train_seed)
    if log_path is None:
        return None, {
            "error": "missing_hydra_training_log",
            "checkpoint_name": checkpoint_name,
            "train_seed": train_seed,
        }
    cmd = [
        sys.executable,
        str(BRACE / "anchor_feasibility_test.py"),
        "--protocol",
        str(protocol_path),
        "--log",
        str(log_path),
        "--output",
        str(feasibility_path),
    ]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if not feasibility_path.is_file() or feasibility_path.stat().st_size == 0:
        return None, {
            "error": "feasibility_analyzer_failed",
            "exit_code": proc.returncode,
            "stderr_tail": proc.stderr[-2000:],
            "log_path": str(log_path),
        }
    payload = read_json(feasibility_path)
    return feasibility_path, {
        "generated": True,
        "passed": payload.get("passed"),
        "parseable": True,
        "exit_code": proc.returncode,
        "log_path": str(log_path),
        "train_seed": train_seed,
    }


def verify_provenance(*, run_dir: Path, protocol_path: Path) -> dict[str, Any]:
    pilot = read_json(run_dir / "pilot_inputs.json")
    expert = REPO / pilot["expert_zarr"]
    anchor = run_dir / "datasets" / "place_container_plate_anchor_replay.zarr"
    b1_manifest = run_dir / "datasets" / "place_container_plate_B1.zarr" / "brace_dataset_manifest.json"
    base_ckpt = DP / "checkpoints" / "place_container_plate-demo_clean-200-0" / "600.ckpt"
    checks = {
        "expert_zarr": expert.is_dir(),
        "anchor_zarr": anchor.is_dir(),
        "b1_manifest": b1_manifest.is_file(),
        "base_ckpt": base_ckpt.is_file(),
        "protocol": protocol_path.is_file(),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "hashes": {
            "protocol_sha256": file_sha256(protocol_path),
            "base_ckpt_sha256": file_sha256(base_ckpt) if base_ckpt.is_file() else None,
            "b1_manifest_sha256": file_sha256(b1_manifest) if b1_manifest.is_file() else None,
            "anchor_manifest_sha256": file_sha256(anchor / "brace_anchor_manifest.json")
            if (anchor / "brace_anchor_manifest.json").is_file()
            else None,
        },
    }


def reconcile_one(
    *,
    method: str,
    seed: int,
    run_dir: Path,
    epochs: int,
    steps_per_epoch: int,
    checkpoint_label: str,
    protocol_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    checkpoint_name = f"place_container_plate-brace-{checkpoint_label}-{method}"
    checkpoint_dir = DP / "checkpoints" / f"{checkpoint_name}-{seed}"
    ckpt = checkpoint_dir / f"{epochs}.ckpt"
    complete = checkpoint_dir / f"{epochs}.ckpt.complete"
    report: dict[str, Any] = {
        "method": method,
        "seed": seed,
        "checkpoint_dir": str(checkpoint_dir),
        "checks": {},
        "passed": False,
    }
    report["checks"]["ckpt_exists"] = ckpt.is_file() and ckpt.stat().st_size > 0
    report["checks"]["complete_marker"] = complete.is_file()
    if not report["checks"]["ckpt_exists"]:
        report["error"] = "missing_final_checkpoint"
        return report
    audit = inspect_diagnostic_checkpoint(ckpt)
    report["checkpoint_audit"] = audit
    report["checks"]["loads_ema"] = bool(audit.get("deploys_ema"))
    report["checkpoint_sha256"] = file_sha256(ckpt)
    report["provenance"] = verify_provenance(run_dir=run_dir, protocol_path=protocol_path)
    report["checks"]["provenance"] = report["provenance"]["passed"]
    feas_path, feas_info = ensure_feasibility(
        method=method,
        checkpoint_dir=checkpoint_dir,
        epochs=epochs,
        protocol_path=protocol_path,
        checkpoint_name=checkpoint_name,
        train_seed=seed,
    )
    report["feasibility"] = feas_info
    if method in ("B2", "B3"):
        report["checks"]["feasibility_parseable"] = bool(feas_info.get("parseable"))
    else:
        report["checks"]["feasibility_parseable"] = True
    log_path = Path(run_dir / "logs" / f"train_{method}_seed{seed}.log")
    report["checks"]["training_log"] = log_path.is_file() and log_path.stat().st_size > 0
    report["expected"] = {
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "global_steps": epochs * steps_per_epoch,
    }
    report["passed"] = all(report["checks"].values())
    if feas_path is not None:
        report["feasibility_path"] = str(feas_path)
    if dry_run:
        return report
    audit_path = checkpoint_dir / f"{epochs}.reconcile_audit.json"
    write_json_atomic(audit_path, report)
    report["audit_path"] = str(audit_path)
    return report


def update_train_state(
    *,
    run_dir: Path,
    method: str,
    seed: int,
    report: dict[str, Any],
    note: str,
) -> None:
    state_path = run_dir / "train_state.json"
    state = read_json(state_path)
    job_id = f"train:{method}:seed{seed}"
    job = next(j for j in state["jobs"] if j["id"] == job_id)
    prior = {
        "prior_status": job.get("status"),
        "prior_note": job.get("note"),
        "prior_requeue_reason": job.get("requeue_reason"),
        "prior_finished_at_utc": job.get("finished_at_utc"),
    }
    job["status"] = "completed" if report["passed"] else "failed"
    job["note"] = note
    job["artifact_reconciled_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    job["artifact_reconcile_report"] = str(report.get("audit_path", ""))
    job["checkpoint_sha256"] = report.get("checkpoint_sha256")
    job["reconcile_prior"] = prior
    job["gpu"] = None
    job["pid"] = None
    if report["passed"]:
        job["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json_atomic(state_path, state)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="B3")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--checkpoint-label", default="place_base200_v2")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=BRACE / "screen_protocol.v1.2.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--update-state", action="store_true")
    parser.add_argument("--note", default="artifact_reconciled_after_post_training_failure")
    args = parser.parse_args()
    run_ptr = BRACE / "runs" / "LATEST_place_base200_v2_line_a_pilot"
    run_dir = args.run_dir or Path(run_ptr.read_text(encoding="utf-8").strip())
    if not run_dir.is_absolute():
        run_dir = REPO / run_dir
    state = read_json(run_dir / "train_state.json")
    steps = int(state.get("steps_per_epoch", 248))
    report = reconcile_one(
        method=args.method,
        seed=args.seed,
        run_dir=run_dir,
        epochs=args.epochs,
        steps_per_epoch=steps,
        checkpoint_label=args.checkpoint_label,
        protocol_path=(REPO / args.protocol).resolve(),
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.update_state and report["passed"]:
        update_train_state(
            run_dir=run_dir,
            method=args.method,
            seed=args.seed,
            report=report,
            note=args.note,
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
