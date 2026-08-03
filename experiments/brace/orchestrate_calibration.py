#!/usr/bin/env python3
"""Orchestrate parallel BRACE anchor calibration jobs across GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.orchestrate import Runner, artifact_complete, prepare_resume_state, utc_id
from experiments.brace.replay_audit import read_json, write_json_atomic


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def base_checkpoint(task: str) -> Path:
    return REPO_ROOT / "policy" / "DP" / "checkpoints" / f"{task}-demo_clean-200-0" / "600.ckpt"


def calibration_command(
    *,
    job_id: str,
    task: str,
    run_label: str,
    protocol: Path,
    jobs_manifest: Path,
    traced_rollout_dir: Path,
    output: Path,
    work_dir: Path,
) -> list[str]:
    return [
        "python",
        str(BRACE_DIR / "anchor_calibration_runner.py"),
        "--protocol",
        str(protocol),
        "--jobs",
        str(jobs_manifest),
        "--job-id",
        job_id,
        "--task",
        task,
        "--run-label",
        run_label,
        "--checkpoint",
        str(base_checkpoint(task)),
        "--traced-rollout-dir",
        str(traced_rollout_dir),
        "--output",
        str(output),
        "--work-dir",
        str(work_dir),
    ]


def create_calibration_state(
    *,
    task: str,
    run_label: str,
    run_dir: Path,
    protocol_path: Path,
    jobs_manifest_path: Path,
    traced_rollout_dir: Path,
    gpus: list[int],
) -> dict[str, Any]:
    manifest = read_json(jobs_manifest_path)
    jobs: dict[str, dict[str, Any]] = {}
    for job in manifest["jobs"]:
        job_id = job["job_id"]
        job_dir = run_dir / job_id
        summary = job_dir / "summary.json"
        jobs[job_id] = {
            "id": job_id,
            "kind": "calibration",
            "status": "pending",
            "dependency": None,
            "artifact": str(summary),
            "log": str(run_dir / "logs" / f"{job_id}.log"),
            "command": calibration_command(
                job_id=job_id,
                task=task,
                run_label=run_label,
                protocol=protocol_path,
                jobs_manifest=jobs_manifest_path,
                traced_rollout_dir=traced_rollout_dir,
                output=summary,
                work_dir=job_dir,
            ),
            "attempts": 0,
            "job_spec": job,
        }
    return {
        "schema_version": 3,
        "stage": "anchor_calibration",
        "status": "running",
        "task": task,
        "run_label": run_label,
        "run_dir": str(run_dir),
        "protocol_path": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "jobs_manifest_path": str(jobs_manifest_path),
        "gpus": gpus,
        "jobs": jobs,
        "events": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def summarize_calibration(state: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for job_id, job in state["jobs"].items():
        artifact = Path(job["artifact"])
        row = {"job_id": job_id, "status": job["status"], "artifact": str(artifact)}
        if artifact.is_file():
            payload = read_json(artifact)
            row.update(
                {
                    "complete": payload.get("complete"),
                    "early_stop_reason": payload.get("early_stop_reason"),
                    "probe_summary": payload.get("probe_summary"),
                    "checkpoint_paths": payload.get("checkpoint_paths"),
                }
            )
        rows.append(row)
    completed = sum(1 for job in state["jobs"].values() if job["status"] == "completed")
    return {
        "schema_version": 3,
        "stage": "anchor_calibration",
        "task": state["task"],
        "run_label": state["run_label"],
        "jobs_total": len(state["jobs"]),
        "jobs_completed": completed,
        "jobs": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=BRACE_DIR / "screen_protocol.v1.3.exploratory_calibration.json")
    parser.add_argument("--jobs", type=Path, default=BRACE_DIR / "calibration_jobs.place_container_plate.v1.json")
    parser.add_argument("--task", default="place_container_plate")
    parser.add_argument("--run-label", default="place_pilot_v2.3")
    parser.add_argument("--traced-rollout-dir", type=Path, default=BRACE_DIR / "rollouts_traced")
    parser.add_argument("--gpus", nargs="*", type=int, default=list(range(8)))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    jobs_manifest_path = args.jobs.resolve()
    if args.state:
        state_path = args.state
        state = read_json(state_path) if state_path.is_file() else None
        run_dir = state_path.parent
        if state is not None and args.resume:
            prepare_resume_state(state)
    elif args.run_dir:
        run_dir = args.run_dir.resolve()
        state_path = run_dir / "state.json"
        state = read_json(state_path) if state_path.is_file() and args.resume else None
        if state is not None and args.resume:
            prepare_resume_state(state)
    else:
        run_dir = BRACE_DIR / "runs" / f"{utc_id()}_anchor_calibration_{args.task}"
        state_path = run_dir / "state.json"
        state = None

    if state is None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        state = create_calibration_state(
            task=args.task,
            run_label=args.run_label,
            run_dir=run_dir,
            protocol_path=protocol_path,
            jobs_manifest_path=jobs_manifest_path,
            traced_rollout_dir=args.traced_rollout_dir.resolve(),
            gpus=args.gpus,
        )
        write_json_atomic(state_path, state)
        (BRACE_DIR / "runs" / "LATEST_anchor_calibration").write_text(str(run_dir), encoding="utf-8")

    if args.dry_run:
        print(json.dumps(state, indent=2))
        return 0

    class CalibrationRunner(Runner):
        def run(self) -> int:
            signal.signal(signal.SIGINT, self.stop)
            signal.signal(signal.SIGTERM, self.stop)
            self.refresh()
            self.save()
            while not self.stopping:
                self.poll()
                failed = [job for job in self.state["jobs"].values() if job["status"] == "failed"]
                if failed:
                    self.state["status"] = "failed"
                    self.save()
                    return 1
                if all(job["status"] == "completed" for job in self.state["jobs"].values()):
                    summary = summarize_calibration(self.state)
                    summary_path = Path(self.state["run_dir"]) / "summary.json"
                    write_json_atomic(summary_path, summary)
                    self.state["status"] = "completed"
                    self.state["summary"] = str(summary_path)
                    self.save()
                    return 0
                free_gpus = [gpu for gpu in self.args.gpus if gpu not in self.running]
                ready = [job for job in self.state["jobs"].values() if self.ready(job)]
                for gpu, job in zip(free_gpus, ready):
                    self.launch(gpu, job)
                if not self.running and not ready:
                    self.state["status"] = "blocked"
                    self.save()
                    return 2
                time.sleep(2)
            self.poll()
            self.state["status"] = "interrupted"
            self.save()
            return 130

    runner = CalibrationRunner(args, state_path, state, read_json(protocol_path))
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
