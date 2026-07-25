#!/usr/bin/env python3
"""Phase 3 CPST scheduler (prep + screen/full/iteration framework).

Stages:
  prep       - build zarr datasets + prefix precision audit
  screen     - short train/eval for A0-A4 (framework; wire to GPUs when ready)
  full       - 760-ep eval for A1 and A4 (mandatory) + optional others
  iteration  - pi_k -> rollout -> CPST -> pi_{k+1} (round 2 framework)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    DEFAULT_GPUS,
    DEFAULT_TASKS,
    FULL_EVAL_REQUIRED,
    MAIN_CANDIDATES,
    PHASE3,
    REPO_ROOT,
    SCREEN_EPOCHS,
    absorption_config,
    atomic_write_json,
    now,
    read_json,
)


def build_prep_jobs(tasks, rollout_dir: Path, require_failures: bool):
    jobs = {}
    for task in tasks:
        audit_artifact = PHASE3 / "prefix_audit" / f"{task}.json"
        for main_id in MAIN_CANDIDATES:
            cfg = absorption_config(main_id)
            dataset_artifact = (
                REPO_ROOT / "policy" / "DP" / "data_phase3" / f"{task}-{cfg['dataset_variant']}.zarr"
            )
            manifest_artifact = dataset_artifact / "phase3_manifest.json"
            job_id = f"prep:dataset:{task}:{main_id}"
            cmd = [
                "python",
                str(PHASE3 / "build_prefix_dataset.py"),
                "--task",
                task,
                "--task-config",
                "demo_clean",
                "--main-id",
                main_id,
                "--rollout-dir",
                str(rollout_dir),
            ]
            if cfg["use_failure_prefix"]:
                cmd.append("--write-candidates")
            jobs[job_id] = {
                "id": job_id,
                "kind": "prep",
                "task": task,
                "main_id": main_id,
                "status": "pending",
                "artifact": str(manifest_artifact),
                "log": str(PHASE3 / "logs" / f"prep_dataset_{task}_{main_id}.log"),
                "command": cmd,
            }
        audit_id = f"prep:prefix_audit:{task}"
        jobs[audit_id] = {
            "id": audit_id,
            "kind": "prep",
            "task": task,
            "status": "pending",
            "artifact": str(audit_artifact),
            "log": str(PHASE3 / "logs" / f"prep_prefix_audit_{task}.log"),
            "command": [
                "python",
                str(PHASE3 / "prefix_audit.py"),
                "--task",
                task,
                "--candidates",
                str(PHASE3 / "prefix_candidates" / f"{task}.json"),
                "--output",
                str(audit_artifact),
            ],
            "optional": not any(absorption_config(m)["use_failure_prefix"] for m in MAIN_CANDIDATES),
        }
    if require_failures:
        for task in tasks:
            failures_dir = rollout_dir / task / "failures"
            jobs[f"prep:check_failures:{task}"] = {
                "id": f"prep:check_failures:{task}",
                "kind": "prep_check",
                "task": task,
                "status": "pending" if failures_dir.is_dir() and any(failures_dir.glob("*.hdf5")) else "blocked",
                "artifact": str(failures_dir / ".ok"),
                "message": (
                    f"Re-collect rollouts with --save-failures if missing: {failures_dir}"
                ),
            }
    return jobs


def create_state(stage: str, tasks, config: dict) -> dict:
    jobs = {}
    if stage == "prep":
        jobs = build_prep_jobs(tasks, Path(config["rollout_dir"]), config.get("require_failures", True))
    elif stage == "screen":
        raise NotImplementedError(
            "Screen stage job wiring is scaffolded in experiments/phase3/README.md. "
            "Use finetune.sh + phase1 eval after prep completes."
        )
    elif stage == "full":
        raise NotImplementedError("Full eval stage: run eval_per_seed.py for A1 and A4 after screen.")
    elif stage == "iteration":
        raise NotImplementedError("Iteration round 2: collect from pi_1 then rebuild datasets with iteration=1.")
    else:
        raise ValueError(f"Unknown stage: {stage}")

    return {
        "version": 1,
        "stage": stage,
        "status": "created",
        "created_at": now(),
        "updated_at": now(),
        "config": config,
        "jobs": jobs,
        "events": [],
    }


def run_prep(state_path: Path, tasks):
    state = read_json(state_path) if state_path.exists() else None
    if state is None:
        config = {
            "stage": "prep",
            "tasks": tasks,
            "rollout_dir": str(REPO_ROOT / "experiments" / "phase1" / "rollouts_200"),
            "require_failures": True,
        }
        state = create_state("prep", tasks, config)
        atomic_write_json(state_path, state)

    import subprocess

    # A state created before failure collection records these checks as
    # "blocked". Refresh them on every resume so the documented
    # collect-rollouts -> prep sequence can actually make progress.
    blocked_checks = []
    for job in state["jobs"].values():
        if job["kind"] != "prep_check":
            continue
        artifact = Path(job["artifact"])
        failures_dir = artifact.parent
        if failures_dir.is_dir() and any(failures_dir.glob("*.hdf5")):
            artifact.touch()
            job["status"] = "completed"
        else:
            job["status"] = "blocked"
            blocked_checks.append(job)

    if blocked_checks:
        state["status"] = "blocked"
        state["updated_at"] = now()
        atomic_write_json(state_path, state)
        for job in blocked_checks:
            print(f"[BLOCKED] {job['id']}: {job.get('message', 'missing prerequisite')}")
        raise SystemExit(2)

    for job in state["jobs"].values():
        if job["status"] in ("completed", "skipped", "blocked"):
            continue
        if job["kind"] == "prep_check":
            print(f"[SKIP] {job['id']}: {job.get('message', 'blocked')}")
            job["status"] = "blocked"
            continue
        log_path = Path(job["log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[RUN] {job['id']}")
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n===== started {now()} =====\n")
            result = subprocess.run(job["command"], cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
        artifact = Path(job["artifact"])
        if result.returncode == 0 and artifact.exists():
            job["status"] = "completed"
            print(f"[OK] {job['id']}")
        elif job.get("optional"):
            job["status"] = "skipped"
            print(f"[SKIP optional] {job['id']}")
        else:
            job["status"] = "failed"
            print(f"[FAIL] {job['id']} exit={result.returncode}")
            state["status"] = "failed"
            atomic_write_json(state_path, state)
            raise SystemExit(1)

    state["status"] = "completed"
    state["completed_at"] = now()
    atomic_write_json(state_path, state)
    print(f"Prep completed: {state_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3 CPST experiment orchestrator.")
    parser.add_argument("stage", choices=("prep", "screen", "full", "iteration", "init"))
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--state-path", type=Path, default=PHASE3 / "run_state_prep.json")
    parser.add_argument("--gpus", nargs="+", type=int, default=list(DEFAULT_GPUS))
    return parser.parse_args()


def main():
    args = parse_args()
    args.state_path.parent.mkdir(parents=True, exist_ok=True)
    if args.stage == "prep":
        run_prep(args.state_path, args.tasks)
        return
    if args.stage == "init":
        for stage in ("prep", "screen", "full"):
            path = PHASE3 / f"run_state_{stage}.json"
            if path.exists():
                print(f"Exists: {path}")
                continue
            config = {
                "stage": stage,
                "tasks": args.tasks,
                "gpus": args.gpus,
                "rollout_dir": str(REPO_ROOT / "experiments" / "phase1" / "rollouts_200"),
                "screen_epochs": list(SCREEN_EPOCHS),
                "full_eval_required": list(FULL_EVAL_REQUIRED),
            }
            try:
                state = create_state(stage, args.tasks, config)
            except NotImplementedError as exc:
                state = {
                    "version": 1,
                    "stage": stage,
                    "status": "scaffold",
                    "created_at": now(),
                    "config": config,
                    "note": str(exc),
                    "jobs": {},
                }
            atomic_write_json(path, state)
            print(f"Initialized {path}")
        return
    raise SystemExit(f"Stage {args.stage} not implemented yet. See experiments/phase3/README.md")


if __name__ == "__main__":
    main()
