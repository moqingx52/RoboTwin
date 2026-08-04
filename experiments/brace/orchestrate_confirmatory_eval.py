#!/usr/bin/env python3
"""Orchestrate confirmatory preservation eval with resume support."""

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
PHASE1_DIR = REPO_ROOT / "experiments" / "phase1"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.confirmatory_preservation_eval import build_candidate_labels
from experiments.brace.confirmatory_common import file_sha256
from experiments.brace.replay_audit import read_json, write_json_atomic


def utc_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def eval_command(
    *,
    label: str,
    task: str,
    output_dir: Path,
    protocol: Path,
    cohort: Path,
    training_run_dir: Path,
    seeds_file: Path,
    workers_per_gpu: int,
) -> list[str]:
    return [
        "python",
        str(BRACE_DIR / "confirmatory_preservation_eval.py"),
        "--task",
        task,
        "--output",
        str(output_dir),
        "--protocol",
        str(protocol),
        "--cohort",
        str(cohort),
        "--training-run-dir",
        str(training_run_dir),
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
    output_dir: Path,
    protocol_path: Path,
    cohort_path: Path,
    training_run_dir: Path,
    seeds_file: Path,
    workers_per_gpu: int,
    gpus: list[int],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    jobs: dict[str, dict[str, Any]] = {}
    for label in build_candidate_labels(protocol):
        variant = "confirm_base_original" if label == "base_original" else f"confirm_{label}"
        jobs[label] = {
            "id": label,
            "kind": "confirmatory_preservation_eval",
            "status": "pending",
            "artifact": str(output_dir / "partials" / f"{label}.json"),
            "candidate_artifact": str(output_dir / task / f"{variant}.json"),
            "log": str(output_dir / "logs" / f"{label}.log"),
            "command": eval_command(
                label=label,
                task=task,
                output_dir=output_dir,
                protocol=protocol_path,
                cohort=cohort_path,
                training_run_dir=training_run_dir,
                seeds_file=seeds_file,
                workers_per_gpu=workers_per_gpu,
            ),
            "attempts": 0,
        }
    return {
        "schema_version": 1,
        "stage": "confirmatory_preservation_eval",
        "status": "running",
        "task": task,
        "output_dir": str(output_dir),
        "training_run_dir": str(training_run_dir),
        "protocol_path": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "cohort_path": str(cohort_path),
        "cohort_sha256": file_sha256(cohort_path),
        "seeds_file": str(seeds_file),
        "seeds_file_sha256": file_sha256(seeds_file),
        "workers_per_gpu": workers_per_gpu,
        "gpus": gpus,
        "jobs": jobs,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def artifact_complete(job: dict[str, Any]) -> bool:
    partial = Path(job["artifact"])
    candidate = Path(job["candidate_artifact"])
    if not (partial.is_file() and candidate.is_file()):
        return False
    try:
        partial_payload = read_json(partial)
        candidate_payload = read_json(candidate)
        return (
            partial_payload.get("label") == job["id"]
            and Path(str(partial_payload.get("eval_output", ""))).resolve() == candidate.resolve()
            and partial_payload.get("eval_sha256") == file_sha256(candidate)
            and bool(candidate_payload.get("progress", {}).get("complete"))
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def isolated_eval_launch(job: dict[str, Any], gpu: int) -> tuple[list[str], dict[str, str]]:
    """Bind one logical eval job to one physical GPU exactly once."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return list(job["command"]), env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="place_container_plate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=BRACE_DIR / "screen_protocol.v1.4.2.confirmatory_preservation.json")
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--training-run-dir", type=Path)
    parser.add_argument("--seeds-file", type=Path, default=BRACE_DIR / "seeds/place_container_plate_confirmatory_v1.4.2_seeds.json")
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument("--gpus", nargs="*", type=int, default=list(range(8)))
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--state", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    protocol_path = args.protocol.resolve()
    protocol = read_json(protocol_path)
    cohort_path = args.cohort.resolve() if args.cohort else Path((BRACE_DIR / "runs/LATEST_preservation_cohort").read_text(encoding="utf-8").strip())
    training_run_dir = args.training_run_dir.resolve() if args.training_run_dir else Path(
        (BRACE_DIR / "runs/LATEST_confirmatory_preservation").read_text(encoding="utf-8").strip()
    )
    seeds_file = args.seeds_file.resolve()
    for required in (protocol_path, cohort_path, training_run_dir, seeds_file):
        if not required.exists():
            raise SystemExit(f"missing confirmatory eval input: {required}")
    cohort = read_json(cohort_path)
    if protocol.get("status") != "frozen" or bool(protocol.get("exploratory", True)):
        raise SystemExit("confirmatory eval requires a frozen non-exploratory protocol")
    if args.task not in protocol.get("tasks", []) or cohort.get("task") != args.task:
        raise SystemExit("task does not match the frozen protocol and cohort")
    if cohort.get("protocol_sha256") != file_sha256(protocol_path):
        raise SystemExit("cohort protocol SHA does not match --protocol")
    if not cohort.get("frozen") or not cohort.get("meets_min_untouched"):
        raise SystemExit("cohort is not frozen and eligible")
    training_state_path = training_run_dir / "state.json"
    if not training_state_path.is_file() or read_json(training_state_path).get("status") != "completed":
        raise SystemExit(f"confirmatory training run is not completed: {training_state_path}")
    state_path = args.state.resolve() if args.state else output_dir / "state.json"
    if args.resume and state_path.is_file():
        state = read_json(state_path)
        expected_state = {
            "protocol_sha256": file_sha256(protocol_path),
            "cohort_sha256": file_sha256(cohort_path),
            "seeds_file_sha256": file_sha256(seeds_file),
            "training_run_dir": str(training_run_dir),
        }
        mismatches = [
            f"{key}={state.get(key)!r}, expected {value!r}"
            for key, value in expected_state.items()
            if state.get(key) != value
        ]
        if mismatches:
            raise SystemExit("refusing incompatible confirmatory eval resume: " + "; ".join(mismatches))
        for job in state["jobs"].values():
            if artifact_complete(job):
                job["status"] = "completed"
            elif job.get("status") in {"running", "failed"}:
                job["status"] = "pending"
                job["attempts"] = 0
                job.pop("exit_code", None)
                job.pop("gpu", None)
                job.pop("pid", None)
        state["status"] = "running"
        write_json_atomic(state_path, state)
    else:
        state = create_state(
            task=args.task,
            output_dir=output_dir,
            protocol_path=protocol_path,
            cohort_path=cohort_path,
            training_run_dir=training_run_dir,
            seeds_file=seeds_file,
            workers_per_gpu=args.workers_per_gpu,
            gpus=args.gpus,
            protocol=protocol,
        )
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
            state["status"] = "failed" if failed else "completed"
            state["completed_at"] = datetime.now(timezone.utc).isoformat()
            write_json_atomic(
                output_dir / "orchestrator_summary.json",
                {
                    "schema_version": 1,
                    "stage": "confirmatory_preservation_eval",
                    "status": state["status"],
                    "jobs_total": len(state["jobs"]),
                    "jobs_completed": sum(
                        1 for job in state["jobs"].values() if job["status"] == "completed"
                    ),
                    "jobs_failed": len(failed),
                    "failed_labels": [job["id"] for job in failed],
                },
            )
            write_json_atomic(state_path, state)
            if failed:
                return 1
            (BRACE_DIR / "runs" / "LATEST_confirmatory_preservation_eval").write_text(
                str(output_dir), encoding="utf-8"
            )
            return 0

        free_gpus = [gpu for gpu in args.gpus if gpu not in running]
        ready = [job for job in state["jobs"].values() if job["status"] == "pending"]
        for gpu, job in zip(free_gpus, ready):
            log_path = Path(job["log"])
            handle = log_path.open("a", encoding="utf-8")
            # The eval child inherits logical cuda:0 and must not rewrite the
            # physical mapping established here.
            command, env = isolated_eval_launch(job, gpu)
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
