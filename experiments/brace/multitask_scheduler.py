#!/usr/bin/env python3
"""Run a frozen BRACE multitask job manifest with resumable GPU scheduling."""

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

from experiments.brace.multitask_preflight import build_preflight
from experiments.brace.multitask_protocol import file_sha256, resolve_repo_path, validate_multitask_protocol
from experiments.brace.replay_audit import read_json, write_json_atomic


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_job_manifest(
    manifest_path: Path,
    protocol_validation: dict[str, Any],
    heldout_tasks: list[str],
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    errors: list[str] = []
    expected_provenance = {
        "protocol_sha256": protocol_validation["protocol_sha256"],
        "task_manifest_sha256": protocol_validation["task_manifest_sha256"],
        "method_freeze_sha256": protocol_validation["method_freeze_sha256"],
    }
    for key, expected in expected_provenance.items():
        if manifest.get(key) != expected:
            errors.append(f"{key} mismatch")
    jobs = manifest.get("jobs", [])
    ids = [str(job.get("id")) for job in jobs]
    if not jobs or len(ids) != len(set(ids)):
        errors.append("jobs must be non-empty with unique ids")
    known = set(ids)
    for job in jobs:
        job_id = str(job.get("id"))
        kind = job.get("kind")
        task = job.get("task")
        command = job.get("command")
        dependencies = job.get("dependencies", [])
        artifact = job.get("artifact")
        if kind not in {"train", "simulator", "cpu"}:
            errors.append(f"invalid kind for {job_id}")
        if task not in heldout_tasks:
            errors.append(f"non-heldout task for {job_id}")
        if not isinstance(command, list) or not command or not all(isinstance(value, str) for value in command):
            errors.append(f"command must be a non-empty argv list for {job_id}")
        if not isinstance(dependencies, list) or any(dep not in known for dep in dependencies):
            errors.append(f"invalid dependencies for {job_id}")
        if not isinstance(artifact, str) or not artifact:
            errors.append(f"missing artifact for {job_id}")
        if kind == "simulator" and int(job.get("workers_per_gpu", 0)) != 3:
            errors.append(f"simulator job {job_id} must declare workers_per_gpu=3")
        if kind == "train" and int(job.get("processes_per_gpu", 0)) != 1:
            errors.append(f"train job {job_id} must declare processes_per_gpu=1")
        if job_id in dependencies:
            errors.append(f"job cannot depend on itself: {job_id}")
    dependencies_by_id = {
        str(job.get("id")): [str(dependency) for dependency in job.get("dependencies", [])]
        for job in jobs
        if isinstance(job.get("dependencies", []), list)
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(job_id: str) -> bool:
        if job_id in visiting:
            return True
        if job_id in visited:
            return False
        visiting.add(job_id)
        if any(dependency in known and visit(dependency) for dependency in dependencies_by_id.get(job_id, [])):
            return True
        visiting.remove(job_id)
        visited.add(job_id)
        return False

    if any(visit(job_id) for job_id in ids if job_id not in visited):
        errors.append("job dependency graph contains a cycle")
    return {"passed": not errors, "errors": errors, "manifest": manifest}


def artifact_complete(job: dict[str, Any]) -> bool:
    artifact = resolve_repo_path(job["artifact"])
    if not artifact.is_file():
        return False
    try:
        payload = read_json(artifact)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(payload.get("complete")) and payload.get("job_id") == job["id"]


def create_state(manifest_path: Path, manifest: dict[str, Any], output_dir: Path, gpus: list[int]) -> dict[str, Any]:
    jobs: dict[str, Any] = {}
    for source in manifest["jobs"]:
        job = dict(source)
        job.update(
            status="completed" if artifact_complete(job) else "pending",
            attempts=0,
            log=str(output_dir / "logs" / f"{job['id'].replace(':', '_')}.log"),
        )
        jobs[job["id"]] = job
    return {
        "schema_version": 1,
        "stage": "brace_multitask_scheduler",
        "status": "running",
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "gpus": gpus,
        "jobs": jobs,
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }


class Scheduler:
    def __init__(self, args: argparse.Namespace, state_path: Path, state: dict[str, Any]):
        self.args = args
        self.state_path = state_path
        self.state = state
        self.running: dict[int, tuple[str, subprocess.Popen[Any], Any]] = {}
        self.stopping = False

    def save(self) -> None:
        self.state["updated_at"] = utc_now()
        write_json_atomic(self.state_path, self.state)

    def ready(self, job: dict[str, Any]) -> bool:
        return job["status"] == "pending" and all(
            self.state["jobs"][dependency]["status"] == "completed"
            for dependency in job.get("dependencies", [])
        )

    def blocked_by_failure(self, job: dict[str, Any]) -> bool:
        return job["status"] == "pending" and any(
            self.state["jobs"][dependency]["status"] in {"failed", "blocked"}
            for dependency in job.get("dependencies", [])
        )

    def propagate_blocked_jobs(self) -> None:
        changed = True
        any_changed = False
        while changed:
            changed = False
            for job in self.state["jobs"].values():
                if self.blocked_by_failure(job):
                    job.update(status="blocked", gpu=None, pid=None)
                    changed = True
                    any_changed = True
        if any_changed:
            self.save()

    def launch(self, gpu: int, job: dict[str, Any]) -> None:
        log_path = Path(job["log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a", encoding="utf-8")
        handle.write("COMMAND " + json.dumps(job["command"]) + "\n")
        handle.flush()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        process = subprocess.Popen(
            job["command"],
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        job.update(status="running", gpu=gpu, pid=process.pid, attempts=int(job["attempts"]) + 1)
        self.running[gpu] = (job["id"], process, handle)
        self.save()

    def poll(self) -> None:
        for gpu, (job_id, process, handle) in list(self.running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            job = self.state["jobs"][job_id]
            if code == 0 and artifact_complete(job):
                status = "completed"
            elif int(job["attempts"]) <= self.args.max_retries:
                status = "pending"
            else:
                status = "failed"
            job.update(status=status, exit_code=code, gpu=None, pid=None)
            del self.running[gpu]
            self.save()

    def stop(self, *_: Any) -> None:
        self.stopping = True
        for _, process, _ in self.running.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)

    def wait_for_running(self) -> None:
        interrupt_deadline = time.monotonic() + 30
        terminate_deadline: float | None = None
        while self.running:
            self.poll()
            now = time.monotonic()
            if self.running and now >= interrupt_deadline and terminate_deadline is None:
                for _, process, _ in self.running.values():
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                terminate_deadline = now + 10
            elif self.running and terminate_deadline is not None and now >= terminate_deadline:
                for _, process, _ in self.running.values():
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                terminate_deadline = float("inf")
            if self.running:
                time.sleep(0.1)

    def run(self) -> int:
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        while not self.stopping:
            self.poll()
            self.propagate_blocked_jobs()
            if all(job["status"] == "completed" for job in self.state["jobs"].values()):
                self.state["status"] = "completed"
                self.state["completed_at"] = utc_now()
                self.save()
                return 0
            free = [gpu for gpu in self.args.gpus if gpu not in self.running]
            ready = [job for job in self.state["jobs"].values() if self.ready(job)]
            for gpu, job in zip(free, ready):
                self.launch(gpu, job)
            if not self.running and not ready:
                failed = any(job["status"] == "failed" for job in self.state["jobs"].values())
                self.state["status"] = "failed" if failed else "blocked"
                self.save()
                return 1 if failed else 2
            time.sleep(2)
        self.wait_for_running()
        self.state["status"] = "interrupted"
        self.save()
        return 130


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=BRACE_DIR / "multitask_protocol.v1.json")
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validation = validate_multitask_protocol(args.protocol, require_method_freeze=True)
    if not validation["passed"]:
        raise SystemExit("invalid or unfrozen multitask protocol: " + "; ".join(validation["errors"]))
    preflight = build_preflight(args.protocol, require_method_freeze=True)
    if not preflight["ready"]:
        raise SystemExit("multitask preflight is not ready")
    heldout = preflight["heldout_tasks"]
    jobs_path = args.jobs.resolve()
    checked = validate_job_manifest(jobs_path, validation, heldout)
    if not checked["passed"]:
        raise SystemExit("invalid multitask job manifest: " + "; ".join(checked["errors"]))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    state_path = args.state.resolve() if args.state else output_dir / "state.json"
    if args.resume:
        if not state_path.is_file():
            raise SystemExit(f"cannot resume missing state: {state_path}")
        state = read_json(state_path)
        if state.get("manifest_sha256") != file_sha256(jobs_path):
            raise SystemExit("cannot resume with a different job manifest")
        for job in state["jobs"].values():
            if artifact_complete(job):
                job["status"] = "completed"
            elif job["status"] in {"running", "failed"}:
                job.update(status="pending", attempts=0, gpu=None, pid=None)
        state["status"] = "running"
    else:
        state = create_state(jobs_path, checked["manifest"], output_dir, args.gpus)
    write_json_atomic(state_path, state)
    if args.dry_run:
        print(json.dumps({"state": str(state_path), "jobs": len(state["jobs"]), "status": state["status"]}, indent=2))
        return 0
    return Scheduler(args, state_path, state).run()


if __name__ == "__main__":
    raise SystemExit(main())
