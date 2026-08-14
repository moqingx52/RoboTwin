#!/usr/bin/env python3
"""Merge per-task replay audit v2 summaries into one combined gate summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.replay_audit import git_commit, read_json, repo_path, write_json_atomic


def merge_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        raise ValueError("need at least one per-task summary")

    tasks: dict[str, Any] = {}
    restore_passed_checks = 0
    restore_total_checks = 0
    restore_failed_checks = 0
    replay_passed_checks = 0
    replay_total_checks = 0
    replay_failed_checks = 0
    preflight_errors: list[str] = []
    complete = True
    per_actor_failure_counts: dict[str, dict[str, int]] = {}

    for summary in summaries:
        for task_name, task_stats in summary.get("tasks", {}).items():
            tasks[task_name] = task_stats
        restore = summary.get("restore_determinism", {})
        replay = summary.get("control_trace_replay", {})
        restore_passed_checks += int(restore.get("passed_checks", 0))
        restore_total_checks += int(restore.get("total_checks", 0))
        restore_failed_checks += int(restore.get("failed_checks", 0))
        replay_passed_checks += int(replay.get("passed_checks", 0))
        replay_total_checks += int(replay.get("total_checks", 0))
        replay_failed_checks += int(replay.get("failed_checks", 0))
        preflight_errors.extend(summary.get("preflight_errors", []))
        complete = complete and bool(summary.get("complete", False))
        for task_name, counts in summary.get("per_actor_failure_counts", {}).items():
            merged = per_actor_failure_counts.setdefault(task_name, {})
            for key, value in counts.items():
                merged[key] = merged.get(key, 0) + int(value)

    restore_pass_rate = restore_passed_checks / restore_total_checks if restore_total_checks else 0.0
    replay_pass_rate = replay_passed_checks / replay_total_checks if replay_total_checks else 0.0
    restore_passed = restore_total_checks > 0 and restore_failed_checks == 0
    replay_required = summaries[0].get("control_trace_replay", {}).get("required_pass_rate", 0.95)
    replay_passed = replay_total_checks > 0 and replay_pass_rate >= float(replay_required)
    per_task_passed = all(task_stats.get("replay_gate_passed", False) for task_stats in tasks.values())
    passed = complete and not preflight_errors and restore_passed and replay_passed and per_task_passed

    total_checks = restore_total_checks + replay_total_checks
    passed_checks = restore_passed_checks + replay_passed_checks
    base = summaries[0]
    return {
        "schema_version": base.get("schema_version", 2),
        "artifacts": base.get("artifacts", {}),
        "complete": complete,
        "passed": passed,
        "preflight_errors": preflight_errors,
        "restore_determinism": {
            "failed_checks": restore_failed_checks,
            "pass_rate": restore_pass_rate,
            "passed": restore_passed,
            "passed_checks": restore_passed_checks,
            "required_pass_rate": summaries[0]
            .get("restore_determinism", {})
            .get("required_pass_rate", 1.0),
            "total_checks": restore_total_checks,
        },
        "control_trace_replay": {
            "failed_checks": replay_failed_checks,
            "pass_rate": replay_pass_rate,
            "passed": replay_passed,
            "passed_checks": replay_passed_checks,
            "required_pass_rate": replay_required,
            "total_checks": replay_total_checks,
        },
        "expected_checks": int(base.get("expected_checks", total_checks)),
        "expected_replay_checks": int(base.get("expected_replay_checks", replay_total_checks)),
        "expected_restore_checks": int(base.get("expected_restore_checks", restore_total_checks)),
        "failed_checks": total_checks - passed_checks,
        "passed_checks": passed_checks,
        "pass_rate": passed_checks / total_checks if total_checks else 0.0,
        "minimum_pass_rate": base.get("minimum_pass_rate", replay_required),
        "per_task_control_trace_replay_minimum_pass_rate": base.get(
            "per_task_control_trace_replay_minimum_pass_rate", replay_required
        ),
        "per_actor_failure_counts": per_actor_failure_counts,
        "tasks": tasks,
        "git_commit": git_commit(),
        "protocol_path": base.get("protocol_path"),
        "protocol_revision": base.get("protocol_revision"),
        "protocol_sha256": base.get("protocol_sha256"),
        "source_summaries": [summary.get("source_summary") for summary in summaries],
        "merge_note": "Combined from per-task immutable audit outputs.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Per-task summary.json paths (repeatable).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/brace/replay_audit_v2/combined_summary.json"),
    )
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    for input_path in args.input:
        path = repo_path(input_path)
        payload = read_json(path)
        payload["source_summary"] = str(path)
        summaries.append(payload)

    merged = merge_summaries(summaries)
    output = repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, merged)
    print(json.dumps({"passed": merged["passed"], "output": str(output)}, indent=2))
    return 0 if merged["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
