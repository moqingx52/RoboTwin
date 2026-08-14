#!/usr/bin/env python3
"""Run Phase 3C behavior eval with one candidate per GPU."""

from __future__ import annotations

import argparse
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

from experiments.brace.anchor_behavior_eval import (
    audit_base_solved_coverage,
    base_solved_seeds_from_manifest,
    build_extra_splits_file,
    build_recommended_candidates,
    find_anchor_manifest,
    select_checkpoint_candidates,
    validate_recommended_candidates,
)
from experiments.brace.replay_audit import read_json, write_json_atomic


def utc_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def behavior_eval_command(
    *,
    label: str,
    task: str,
    calibration_run_dir: Path,
    output_dir: Path,
    protocol: Path,
    seeds_file: Path,
    workers_per_gpu: int,
) -> list[str]:
    return [
        "python",
        str(BRACE_DIR / "anchor_behavior_eval.py"),
        "--task",
        task,
        "--calibration-run-dir",
        str(calibration_run_dir),
        "--output",
        str(output_dir),
        "--protocol",
        str(protocol),
        "--seeds-file",
        str(seeds_file),
        "--label",
        label,
        "--workers-per-gpu",
        str(workers_per_gpu),
    ]


def create_state(
    *,
    task: str,
    calibration_run_dir: Path,
    output_dir: Path,
    protocol_path: Path,
    seeds_file: Path,
    workers_per_gpu: int,
    gpus: list[int],
) -> dict[str, Any]:
    candidates = select_checkpoint_candidates(calibration_run_dir, task=task)
    validate_recommended_candidates(candidates)
    jobs: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        label = candidate["label"]
        jobs[label] = {
            "id": label,
            "kind": "behavior_eval",
            "status": "pending",
            "dependency": None,
            "artifact": str(output_dir / "partials" / f"{label}.json"),
            "candidate_artifact": str(output_dir / task / f"calib_{label}.json"),
            "log": str(output_dir / "logs" / f"{label}.log"),
            "command": behavior_eval_command(
                label=label,
                task=task,
                calibration_run_dir=calibration_run_dir,
                output_dir=output_dir,
                protocol=protocol_path,
                seeds_file=seeds_file,
                workers_per_gpu=workers_per_gpu,
            ),
            "attempts": 0,
            "candidate": candidate,
        }
    return {
        "schema_version": 2,
        "stage": "anchor_behavior_eval",
        "status": "running",
        "task": task,
        "calibration_run_dir": str(calibration_run_dir),
        "output_dir": str(output_dir),
        "workers_per_gpu": workers_per_gpu,
        "gpus": gpus,
        "jobs": jobs,
        "events": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def artifact_complete(job: dict[str, Any]) -> bool:
    partial_path = Path(job["artifact"])
    candidate_path = Path(job["candidate_artifact"])
    return (
        partial_path.is_file()
        and partial_path.stat().st_size > 0
        and candidate_path.is_file()
        and candidate_path.stat().st_size > 0
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="place_container_plate")
    parser.add_argument("--calibration-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=BRACE_DIR / "screen_protocol.v1.3.exploratory_calibration.json")
    parser.add_argument("--seeds-file", type=Path, default=REPO_ROOT / "experiments" / "phase1" / "seeds" / "place_container_plate_seeds.json")
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument("--gpus", nargs="*", type=int, default=list(range(6)))
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.gpus:
        raise SystemExit("--gpus must contain at least one physical GPU id")
    if len(set(args.gpus)) != len(args.gpus):
        raise SystemExit(f"--gpus contains duplicate physical GPU ids: {args.gpus}")
    if args.workers_per_gpu < 1:
        raise SystemExit("--workers-per-gpu must be >= 1")
    for required in (args.calibration_run_dir, args.protocol, args.seeds_file):
        if not required.exists():
            raise SystemExit(f"missing required behavior-eval input: {required}")
    hard_seeds_file = (
        REPO_ROOT / "experiments" / "phase1" / "eval_results_200" / "hard_eval_seeds" / f"{args.task}.json"
    )
    if not hard_seeds_file.is_file():
        raise SystemExit(f"missing required behavior-eval input: {hard_seeds_file}")

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    manifest_path = find_anchor_manifest(args.calibration_run_dir.resolve(), task=args.task)
    base_solved = base_solved_seeds_from_manifest(manifest_path)
    audit = audit_base_solved_coverage(base_solved_seeds=base_solved, seeds_file=args.seeds_file.resolve())
    extra_splits = build_extra_splits_file(base_solved_seeds=base_solved, audit=audit, work_dir=output_dir)
    write_json_atomic(output_dir / "base_solved_coverage_audit.json", audit)

    state = create_state(
        task=args.task,
        calibration_run_dir=args.calibration_run_dir.resolve(),
        output_dir=output_dir,
        protocol_path=args.protocol.resolve(),
        seeds_file=args.seeds_file.resolve(),
        workers_per_gpu=args.workers_per_gpu,
        gpus=args.gpus,
    )
    state_path = output_dir / "state.json"
    write_json_atomic(state_path, state)
    if args.dry_run:
        print(json.dumps(state, indent=2))
        return 0

    running: dict[int, tuple[str, subprocess.Popen, Any]] = {}
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True
        for _, process, _ in running.values():
            if process.poll() is None:
                process.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    while not stopping:
        for gpu, (job_id, process, handle) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            job = state["jobs"][job_id]
            if code == 0 and artifact_complete(job):
                job["status"] = "completed"
            elif int(job["attempts"]) <= args.max_retries:
                job["status"] = "pending"
            else:
                job["status"] = "failed"
            job["exit_code"] = code
            del running[gpu]
            write_json_atomic(state_path, state)

        terminal = all(job["status"] in ("completed", "failed") for job in state["jobs"].values())
        if terminal:
            failed = [job for job in state["jobs"].values() if job["status"] == "failed"]
            if failed:
                state["status"] = "failed"
                write_json_atomic(state_path, state)
                write_json_atomic(
                    output_dir / "orchestrator_summary.json",
                    {
                        "schema_version": 2,
                        "stage": "anchor_behavior_eval",
                        "status": "failed",
                        "jobs_total": len(state["jobs"]),
                        "jobs_completed": sum(
                            1 for job in state["jobs"].values() if job["status"] == "completed"
                        ),
                        "jobs_failed": len(failed),
                        "failed_labels": [job["id"] for job in failed],
                    },
                )
                return 1

            aggregate = subprocess.run(
                [
                    "python",
                    str(BRACE_DIR / "anchor_behavior_eval.py"),
                    "--task",
                    args.task,
                    "--calibration-run-dir",
                    str(args.calibration_run_dir.resolve()),
                    "--output",
                    str(output_dir),
                    "--protocol",
                    str(args.protocol.resolve()),
                    "--seeds-file",
                    str(args.seeds_file.resolve()),
                    "--aggregate-only",
                ],
                cwd=REPO_ROOT,
                check=False,
            )
            if aggregate.returncode != 0 or not (output_dir / "summary.json").is_file():
                state["status"] = "failed"
                state["aggregate_exit_code"] = aggregate.returncode
                write_json_atomic(state_path, state)
                return 1

            summary = read_json(output_dir / "summary.json")
            state["status"] = "completed"
            state["summary"] = str(output_dir / "summary.json")
            write_json_atomic(state_path, state)
            write_json_atomic(
                output_dir / "orchestrator_summary.json",
                {
                    "schema_version": 2,
                    "stage": "anchor_behavior_eval",
                    "jobs_total": len(state["jobs"]),
                    "jobs_completed": sum(1 for job in state["jobs"].values() if job["status"] == "completed"),
                    "jobs_failed": 0,
                    "behavior_summary": summary,
                },
            )
            (BRACE_DIR / "runs" / "LATEST_anchor_behavior_eval").write_text(str(output_dir), encoding="utf-8")
            return 1 if failed else 0

        free_gpus = [gpu for gpu in args.gpus if gpu not in running]
        ready = [job for job in state["jobs"].values() if job["status"] == "pending"]
        for gpu, job in zip(free_gpus, ready):
            log_path = Path(job["log"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            # anchor_behavior_eval owns the environment passed to run_eval_group, so pass
            # the physical id here. Passing logical 0 would overwrite this process's
            # CUDA_VISIBLE_DEVICES mask and pile every candidate onto physical GPU 0.
            command = list(job["command"]) + ["--gpu", str(gpu)]
            handle.write("COMMAND " + json.dumps(command) + "\n")
            handle.flush()
            process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            job["status"] = "running"
            job["attempts"] = int(job["attempts"]) + 1
            job["gpu"] = gpu
            running[gpu] = (job["id"], process, handle)
            write_json_atomic(state_path, state)
        if not running and not ready:
            break
        time.sleep(2)

    write_json_atomic(state_path, state)
    return 130


if __name__ == "__main__":
    raise SystemExit(main())
