#!/usr/bin/env python3
"""Recover Base200 Line A adaptation eval after merge/zombie stall.

Does not rebuild eval_state.json. Merges completed easy shards, validates them,
and atomically resets status=running jobs to pending so a restarted scheduler
can skip complete easy variants and continue to hard.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
BRACE = REPO / "experiments" / "brace"
PHASE1 = REPO / "experiments" / "phase1"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(PHASE1) not in sys.path:
    sys.path.insert(0, str(PHASE1))

from experiments.brace.replay_audit import read_json, write_json_atomic

TASK = "place_container_plate"
NUM_SHARDS = 3


def work_item_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row["split"]), int(row["env_seed"]), int(row["repeat"]))


def validate_shards(task_dir: Path, variant: str) -> dict[str, Any]:
    paths = sorted(task_dir.glob(f"{variant}_shard_*_of_{NUM_SHARDS:02d}.json"))
    errors: list[str] = []
    if len(paths) != NUM_SHARDS:
        errors.append(f"expected {NUM_SHARDS} shards, found {len(paths)}")
    seen: set[tuple[str, int, int]] = set()
    for path in paths:
        payload = read_json(path)
        progress = payload.get("progress") or {}
        if not progress.get("complete"):
            errors.append(f"{path.name} progress.complete is not true")
        if int(progress.get("num_shards", -1)) != NUM_SHARDS:
            errors.append(f"{path.name} num_shards={progress.get('num_shards')}")
        for row in payload.get("rows") or []:
            key = work_item_key(row)
            if key in seen:
                errors.append(f"duplicate work item {key}")
            seen.add(key)
    return {"variant": variant, "n_shards": len(paths), "n_work_items": len(seen), "errors": errors}


def merge_variant(*, output_dir: Path, variant: str, task_config: str) -> dict[str, Any]:
    check = validate_shards(output_dir / TASK, variant)
    if check["errors"]:
        return {**check, "merged": False}
    cmd = [
        sys.executable,
        str(PHASE1 / "merge_eval_shards.py"),
        "--task",
        TASK,
        "--task-config",
        task_config,
        "--variant",
        variant,
        "--output-dir",
        str(output_dir),
        "--num-shards",
        str(NUM_SHARDS),
    ]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    merged_path = output_dir / TASK / f"{variant}.json"
    ok = proc.returncode == 0 and merged_path.is_file()
    complete = False
    if ok:
        payload = read_json(merged_path)
        complete = bool((payload.get("progress") or {}).get("complete"))
        if not complete:
            check["errors"].append("merged progress.complete is not true")
    elif proc.returncode != 0:
        check["errors"].append(f"merge exit={proc.returncode} stderr={proc.stderr[-1000:]}")
    return {
        **check,
        "merged": ok and complete,
        "merged_path": str(merged_path),
        "stdout": proc.stdout.strip(),
    }


def reset_running_jobs(state: dict[str, Any]) -> list[str]:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    reset: list[str] = []
    for job in state.get("jobs") or []:
        if job.get("status") != "running":
            continue
        job["status"] = "pending"
        job.pop("pid", None)
        job.pop("gpu", None)
        job["recovery_note"] = {
            "reclaimed_from": "running",
            "at_utc": now,
            "reason": "merge_zombie_stall_reset_before_resume",
        }
        reset.append(str(job.get("id")))
    return reset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Pilot run dir (default: LATEST_place_base200_v2_line_a_pilot).",
    )
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-state-reset", action="store_true")
    args = parser.parse_args()
    if args.run_dir is None:
        ptr = BRACE / "runs/LATEST_place_base200_v2_line_a_pilot"
        run_dir = Path(ptr.read_text(encoding="utf-8").strip())
        if not run_dir.is_absolute():
            run_dir = REPO / run_dir
    else:
        run_dir = args.run_dir if args.run_dir.is_absolute() else REPO / args.run_dir

    state_path = run_dir / "eval_state.json"
    state = read_json(state_path)
    output_dir = Path(state.get("output_dir") or (run_dir / "eval"))
    running_jobs = [j for j in state.get("jobs") or [] if j.get("status") == "running"]
    merge_reports = []
    if not args.skip_merge:
        for job in running_jobs:
            variants = job.get("variants") or []
            if not variants:
                continue
            easy = variants[0]
            merge_reports.append(
                merge_variant(
                    output_dir=output_dir,
                    variant=easy["variant"],
                    task_config=easy.get("task_config") or "demo_clean",
                )
            )
    merge_failed = [r for r in merge_reports if not r.get("merged")]
    reset_ids: list[str] = []
    if not args.skip_state_reset:
        if merge_failed:
            raise SystemExit("refusing to reset state while merge validation failed: " + json.dumps(merge_failed))
        reset_ids = reset_running_jobs(state)
        write_json_atomic(state_path, state)

    report = {
        "run_dir": str(run_dir),
        "state_path": str(state_path),
        "n_running_before": len(running_jobs),
        "reset_job_ids": reset_ids,
        "merge_reports": merge_reports,
        "status_after": {
            status: sum(1 for j in state.get("jobs") or [] if j.get("status") == status)
            for status in sorted({j.get("status") for j in state.get("jobs") or []})
        },
    }
    out = run_dir / "eval_recovery_report.json"
    write_json_atomic(out, report)
    print(json.dumps(report, indent=2))
    return 0 if not merge_failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
