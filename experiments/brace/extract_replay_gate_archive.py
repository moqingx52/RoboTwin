#!/usr/bin/env python3
"""Extract a per-task replay audit v2 summary into an archive gate bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.replay_audit import read_json, repo_path, write_json_atomic

TASK_TO_ARCHIVE = {
    "place_container_plate": "experiments/brace/archive/replay_audit_v2_place_v2.3_gate",
    "dump_bin_bigbin": "experiments/brace/archive/replay_audit_v2_dump_v2.3_gate",
}


def summarize_check_type(checks: list[dict[str, Any]], check_type: str) -> dict[str, Any]:
    rows = [row for row in checks if row.get("check_type") == check_type]
    passed_checks = sum(1 for row in rows if row.get("passed"))
    total_checks = len(rows)
    failed_checks = total_checks - passed_checks
    return {
        "failed_checks": failed_checks,
        "pass_rate": passed_checks / total_checks if total_checks else 0.0,
        "passed_checks": passed_checks,
        "total_checks": total_checks,
    }


def load_checks(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_task_summary(source: dict[str, Any], task: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    task_stats = source.get("tasks", {}).get(task)
    if task_stats is None:
        raise ValueError(f"task not found in summary: {task}")

    task_checks = [row for row in checks if row.get("task") == task]
    restore_stats = summarize_check_type(task_checks, "restore_determinism")
    replay_stats = summarize_check_type(task_checks, "control_trace_replay")
    restore_required = float(
        source.get("restore_determinism", {}).get("required_pass_rate", 1.0)
    )
    replay_required = float(
        source.get("control_trace_replay", {}).get("required_pass_rate", 0.95)
    )
    restore_passed = (
        restore_stats["total_checks"] > 0
        and restore_stats["failed_checks"] == 0
        and restore_stats["pass_rate"] >= restore_required
    )
    replay_passed = (
        replay_stats["total_checks"] > 0
        and replay_stats["pass_rate"] >= replay_required
        and bool(task_stats.get("replay_gate_passed", replay_stats["pass_rate"] >= replay_required))
    )
    total_checks = restore_stats["total_checks"] + replay_stats["total_checks"]
    passed_checks = restore_stats["passed_checks"] + replay_stats["passed_checks"]
    passed = (
        bool(source.get("complete", False))
        and not source.get("preflight_errors")
        and restore_passed
        and replay_passed
        and bool(task_stats.get("replay_gate_passed", replay_passed))
    )

    per_actor = {
        task: dict(source.get("per_actor_failure_counts", {}).get(task, {}))
    } if task in source.get("per_actor_failure_counts", {}) else {}

    return {
        "schema_version": source.get("schema_version", 2),
        "artifacts": source.get("artifacts", {}),
        "complete": bool(source.get("complete", False)),
        "passed": passed,
        "preflight_errors": list(source.get("preflight_errors", [])),
        "restore_determinism": {
            **restore_stats,
            "passed": restore_passed,
            "required_pass_rate": restore_required,
        },
        "control_trace_replay": {
            **replay_stats,
            "passed": replay_passed,
            "required_pass_rate": replay_required,
        },
        "expected_checks": total_checks,
        "expected_replay_checks": replay_stats["total_checks"],
        "expected_restore_checks": restore_stats["total_checks"],
        "failed_checks": total_checks - passed_checks,
        "passed_checks": passed_checks,
        "pass_rate": passed_checks / total_checks if total_checks else 0.0,
        "minimum_pass_rate": replay_required,
        "per_task_control_trace_replay_minimum_pass_rate": source.get(
            "per_task_control_trace_replay_minimum_pass_rate", replay_required
        ),
        "per_actor_failure_counts": per_actor,
        "protocol_path": source.get("protocol_path"),
        "protocol_revision": source.get("protocol_revision"),
        "protocol_sha256": source.get("protocol_sha256"),
        "git_commit": source.get("git_commit"),
        "tasks": {task: task_stats},
        "total_checks": total_checks,
        "source_summary": source.get("source_summary"),
        "archive_note": f"Per-task gate archive extracted for {task}.",
    }


def resolve_source(task: str, audit_root: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    per_task = audit_root / task / "summary.json"
    if per_task.is_file():
        summary = read_json(per_task)
        checks = load_checks(audit_root / task / "checks.jsonl")
        summary["source_summary"] = str(per_task)
        return per_task, summary, checks

    combined = audit_root / "summary.json"
    if not combined.is_file():
        raise FileNotFoundError(f"no replay audit summary under {audit_root}")
    summary = read_json(combined)
    checks = load_checks(audit_root / "checks.jsonl")
    summary["source_summary"] = str(combined)
    if task not in summary.get("tasks", {}):
        raise ValueError(f"{task} not present in {combined}")
    return combined, summary, checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASK_TO_ARCHIVE))
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=Path("experiments/brace/replay_audit_v2"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    audit_root = repo_path(args.audit_root)
    archive_dir = repo_path(args.output_dir or TASK_TO_ARCHIVE[args.task])
    archive_dir.mkdir(parents=True, exist_ok=True)

    _, source_summary, checks = resolve_source(args.task, audit_root)
    payload = extract_task_summary(source_summary, args.task, checks)
    output = archive_dir / "summary.json"
    write_json_atomic(output, payload)
    print(json.dumps({"task": args.task, "passed": payload["passed"], "output": str(output)}, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
