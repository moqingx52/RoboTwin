#!/usr/bin/env python3
"""Rebuild a replay-audit v2 gate summary from recovered checks.jsonl.

This is intentionally conservative: facts not derivable from checks are left
null, and recovery provenance is embedded in the output.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.replay_audit import file_sha256, read_json, repo_path, write_json_atomic
from experiments.brace.replay_audit_v2 import (
    METRIC_TO_THRESHOLD,
    evaluate_replay_gate,
    evaluate_task_replay_gate,
    replay_gate_requirements,
    summarize_per_actor_failures,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rebuild_summary(*, checks_path: Path, protocol_path: Path, task: str) -> dict[str, Any]:
    checks = read_jsonl(checks_path)
    protocol = read_json(protocol_path)
    task_checks = [row for row in checks if row.get("task") == task]
    if not task_checks:
        raise ValueError(f"no checks for {task}: {checks_path}")
    unexpected = sorted({str(row.get("task")) for row in checks} - {task})
    if unexpected:
        raise ValueError(f"checks contain other tasks: {unexpected}")

    expected_each = int(protocol["replay_audit"]["trajectories_per_task"]) * int(
        protocol["replay_audit"]["snapshots_per_trajectory"]
    )
    expected_checks = expected_each * 2
    complete = len(task_checks) == expected_checks
    requirements = replay_gate_requirements(protocol)
    task_gate = evaluate_task_replay_gate(checks, task, requirements)
    gate = evaluate_replay_gate(checks, [task], protocol, complete=complete, preflight_errors=[])

    trajectory_keys = {
        (int(row["env_seed"]), int(row["rollout_id"]), bool(row.get("success")))
        for row in task_checks
    }
    outcome_counts = Counter(success for _, _, success in trajectory_keys)
    passed_checks = sum(bool(row.get("passed")) for row in task_checks)
    metric_errors = {
        metric: {
            "maximum": max(values),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
        }
        for metric in METRIC_TO_THRESHOLD
        if (
            values := [
                float(row["errors"][metric])
                for row in task_checks
                if row.get("errors") and metric in row["errors"]
            ]
        )
    }
    task_summary = {
        "available_trajectories": None,
        "requested_trajectories": int(protocol["replay_audit"]["trajectories_per_task"]),
        "selected_trajectories": len(trajectory_keys),
        "selected_successes": int(outcome_counts[True]),
        "selected_failures": int(outcome_counts[False]),
        "snapshots_per_trajectory": int(protocol["replay_audit"]["snapshots_per_trajectory"]),
        "preflight_errors": [],
        "total_checks": len(task_checks),
        "passed_checks": passed_checks,
        "pass_rate": passed_checks / len(task_checks),
        "restore_determinism": task_gate["restore_determinism"],
        "control_trace_replay": task_gate["control_trace_replay"],
        "replay_gate_passed": task_gate["replay_gate_passed"],
    }
    return {
        "schema_version": 2,
        "protocol_revision": protocol.get("protocol_revision"),
        "passed": gate["passed"],
        "complete": complete,
        "pass_rate": gate["mixed_pass_rate"],
        "minimum_pass_rate": float(protocol["replay_gate"].get("minimum_pass_rate", 0.95)),
        "passed_checks": gate["passed_checks"],
        "total_checks": gate["total_checks"],
        "expected_checks": expected_checks,
        "expected_restore_checks": expected_each,
        "expected_replay_checks": expected_each,
        "failed_checks": gate["failed_checks"],
        "restore_determinism": gate["restore_determinism"],
        "control_trace_replay": gate["control_trace_replay"],
        "per_task_control_trace_replay_minimum_pass_rate": gate[
            "per_task_control_trace_replay_minimum_pass_rate"
        ],
        "protocol_path": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "git_commit": None,
        "thresholds": {
            metric: float(protocol["replay_gate"][protocol_key])
            for metric, protocol_key in METRIC_TO_THRESHOLD.items()
        },
        "metric_errors": metric_errors,
        "per_actor_failure_counts": summarize_per_actor_failures(task_checks),
        "tasks": {task: task_summary},
        "preflight_errors": [],
        "artifacts": {
            "checks": "checks.jsonl",
            "failures": "failures.jsonl",
            "diagnostics": "diagnostics.jsonl",
        },
        "recovery_provenance": {
            "method": "rebuild_from_checks",
            "checks_path": str(checks_path),
            "checks_sha256": file_sha256(checks_path),
            "note": "Original summary was unavailable or mismatched; no unavailable metadata was imputed.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = rebuild_summary(
        checks_path=repo_path(args.checks),
        protocol_path=repo_path(args.protocol),
        task=args.task,
    )
    write_json_atomic(repo_path(args.output), payload)
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
