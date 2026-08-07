#!/usr/bin/env python3
"""Aggregate per-task seed feasibility outcomes into a batch summary (schema v2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.multitask_protocol import file_sha256
from experiments.brace.replay_audit import read_json, write_json_atomic
from experiments.brace.seed_feasibility import (
    FEASIBILITY_EVIDENCE_SCHEMA_VERSION,
    TASK_STATUS_PASSED,
    TASK_STATUS_PENDING,
    validate_feasibility_evidence,
)


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def build_batch_summary(run_dir: Path, tasks: list[str], *, require_task_status: bool = True) -> dict:
    task_rows: dict[str, dict] = {}
    unresolved = 0
    for task in tasks:
        # Preference: supplemented evidence > derived v2 > plain v1/v2 run output.
        evidence_path = run_dir / f"{task}_supplemented_feasibility.json"
        if not evidence_path.is_file():
            evidence_path = run_dir / f"{task}_feasibility_v2.json"
        if not evidence_path.is_file():
            evidence_path = run_dir / f"{task}_feasibility.json"
        if not evidence_path.is_file():
            task_rows[task] = {
                "task_status": TASK_STATUS_PENDING,
                "passed": False,
                "evidence_path": None,
                "evidence_sha256": None,
            }
            unresolved += 1
            continue
        evidence = read_json(evidence_path)
        status = evidence.get("task_status")
        validation_errors = validate_feasibility_evidence(evidence)
        if require_task_status and status is None:
            validation_errors.append("evidence missing task_status; run upgrade_archived_feasibility_v2.py")
        if validation_errors:
            task_rows[task] = {
                "task_status": TASK_STATUS_PENDING,
                "passed": False,
                "evidence_path": _repo_relative(evidence_path),
                "evidence_sha256": file_sha256(evidence_path),
                "invalid_evidence": validation_errors,
            }
            unresolved += 1
            continue
        task_rows[task] = {
            "task_status": status,
            "passed": status == TASK_STATUS_PASSED,
            "solvable_count_in_candidates": evidence.get("solvable_count_in_candidates"),
            "expert_exception_count": evidence.get("expert_exception_count"),
            "seed_not_solvable_count": evidence.get("seed_not_solvable_count"),
            "shard_count": evidence.get("shard_count"),
            "gpus": evidence.get("gpus"),
            "supplement_used": bool(evidence.get("supplement", {}).get("used")),
            "evidence_path": _repo_relative(evidence_path),
            "evidence_sha256": file_sha256(evidence_path),
            "evidence_schema_version": evidence.get("schema_version"),
        }
        if status != TASK_STATUS_PASSED:
            unresolved += 1
    return {
        "schema_version": 2,
        "stage": "seed_feasibility_batch",
        "run_dir": _repo_relative(run_dir),
        "evidence_schema_version": FEASIBILITY_EVIDENCE_SCHEMA_VERSION,
        "tasks": task_rows,
        "task_count": len(tasks),
        "passed_count": sum(1 for row in task_rows.values() if row["passed"]),
        "unresolved_count": unresolved,
        "all_passed": unresolved == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-task-status",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reject evidence files that lack a task_status field",
    )
    args = parser.parse_args()
    summary = build_batch_summary(args.run_dir.resolve(), args.tasks, require_task_status=args.require_task_status)
    output = args.output or (args.run_dir / "batch_summary.json")
    write_json_atomic(output, summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
