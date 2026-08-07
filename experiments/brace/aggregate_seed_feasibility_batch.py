#!/usr/bin/env python3
"""Aggregate per-task seed feasibility outcomes into a batch summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.replay_audit import read_json, write_json_atomic
from experiments.brace.seed_feasibility import TASK_STATUS_PASSED


def build_batch_summary(run_dir: Path, tasks: list[str]) -> dict:
    task_rows: dict[str, dict] = {}
    unresolved = 0
    for task in tasks:
        evidence_path = run_dir / f"{task}_feasibility.json"
        if not evidence_path.is_file():
            task_rows[task] = {
                "task_status": "pending",
                "passed": False,
                "evidence_path": None,
            }
            unresolved += 1
            continue
        evidence = read_json(evidence_path)
        status = evidence.get("task_status", "pending")
        task_rows[task] = {
            "task_status": status,
            "passed": status == TASK_STATUS_PASSED,
            "solvable_count_in_candidates": evidence.get("solvable_count_in_candidates"),
            "expert_exception_count": evidence.get("expert_exception_count"),
            "seed_not_solvable_count": evidence.get("seed_not_solvable_count"),
            "evidence_path": str(evidence_path),
        }
        if status != TASK_STATUS_PASSED:
            unresolved += 1
    return {
        "schema_version": 1,
        "stage": "seed_feasibility_batch",
        "run_dir": str(run_dir),
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
    args = parser.parse_args()
    summary = build_batch_summary(args.run_dir.resolve(), args.tasks)
    output = args.output or (args.run_dir / "batch_summary.json")
    write_json_atomic(output, summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
