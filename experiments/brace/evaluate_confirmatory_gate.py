#!/usr/bin/env python3
"""Merge pilot + confirmatory branch summaries and evaluate Track B gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.replay_audit import git_commit, read_json, repo_path, write_json_atomic


def load_summary(path: Path) -> dict[str, Any]:
    return read_json(repo_path(path))


def merge_task_points(
    pilot: dict[str, Any],
    confirm: dict[str, Any],
    task: str,
) -> dict[str, Any]:
    pilot_task = pilot.get("tasks", {}).get(task, {})
    confirm_task = confirm.get("tasks", {}).get(task, {})
    pilot_points = list(pilot_task.get("points", []))
    confirm_points = list(confirm_task.get("points", []))
    all_points = pilot_points + confirm_points

    accepted = [point for point in all_points if point.get("accepted")]
    seeds = {int(point["env_seed"]) for point in all_points}
    accepted_by_seed = Counter(int(point["env_seed"]) for point in accepted)

    matched_lift_values = [
        float(point["candidate_success_rate"]) - float(point["control_success_rate"]) for point in all_points
    ]
    matched_lift = sum(matched_lift_values) / len(matched_lift_values) if matched_lift_values else 0.0

    recovery_seeds = {
        int(point["env_seed"])
        for point in all_points
        if point.get("point_type") != "random_negative_control" and float(point.get("candidate_success_rate", 0)) > 0
    }
    recovery_fraction = len(recovery_seeds) / len(seeds) if seeds else 0.0

    max_seed_share = (
        max(accepted_by_seed.values()) / len(accepted) if accepted else 0.0
    )

    checks = {
        "harness_valid": bool(pilot.get("harness_valid")) and bool(confirm.get("harness_valid")),
        "lift_positive": matched_lift > 0,
        "lift_or_recovery_gate": matched_lift >= 0.20 or recovery_fraction >= 0.20,
        "min_seeds": len(seeds) >= 10,
        "min_points": len(all_points) >= 30,
        "max_seed_accepted_share_le_20pct": max_seed_share <= 0.20,
    }
    passed = all(checks.values())

    return {
        "task": task,
        "passed": passed,
        "checks": checks,
        "pilot_points": len(pilot_points),
        "confirm_points": len(confirm_points),
        "total_points": len(all_points),
        "accepted_points": len(accepted),
        "unique_seeds": len(seeds),
        "matched_success_lift": matched_lift,
        "recovery_seed_fraction": recovery_fraction,
        "max_accepted_share_by_seed": max_seed_share,
        "accepted_by_seed": dict(sorted(accepted_by_seed.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-summary", type=Path, required=True)
    parser.add_argument("--confirm-summary", type=Path, required=True)
    parser.add_argument("--task", default="place_container_plate")
    parser.add_argument("--output", type=Path, default=Path("experiments/brace/branches_confirm/merged_gate.json"))
    args = parser.parse_args()

    pilot = load_summary(args.pilot_summary)
    confirm = load_summary(args.confirm_summary)
    merged = merge_task_points(pilot, confirm, args.task)
    payload = {
        "schema_version": 1,
        "git_commit": git_commit(),
        "pilot_summary": str(repo_path(args.pilot_summary)),
        "confirm_summary": str(repo_path(args.confirm_summary)),
        **merged,
    }
    output = repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, payload)
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
