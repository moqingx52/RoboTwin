#!/usr/bin/env python3
"""Merge pilot + confirmatory branch summaries and evaluate Track B gate."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.replay_audit import git_commit, read_json, repo_path, write_json_atomic


def load_summary(path: Path) -> dict[str, Any]:
    return read_json(repo_path(path))


def point_lift(point: dict[str, Any]) -> float:
    return float(point["candidate_success_rate"]) - float(point["control_success_rate"])


def seed_level_lifts(points: list[dict[str, Any]]) -> dict[int, float]:
    by_seed: dict[int, list[float]] = defaultdict(list)
    for point in points:
        by_seed[int(point["env_seed"])].append(point_lift(point))
    return {seed: sum(values) / len(values) for seed, values in sorted(by_seed.items())}


def bootstrap_ci(
    seed_lifts: dict[int, float],
    *,
    samples: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    if not seed_lifts:
        return 0.0, 0.0
    values = list(seed_lifts.values())
    if len(values) == 1:
        value = values[0]
        return value, value
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(samples):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(draw) / n)
    means.sort()
    low_index = int((alpha / 2) * len(means))
    high_index = max(low_index, int((1 - alpha / 2) * len(means)) - 1)
    return means[low_index], means[high_index]


def leave_one_seed_out(seed_lifts: dict[int, float]) -> dict[str, float]:
    if len(seed_lifts) <= 1:
        return {}
    overall = sum(seed_lifts.values()) / len(seed_lifts)
    result: dict[str, float] = {}
    for dropped_seed in seed_lifts:
        remaining = [value for seed, value in seed_lifts.items() if seed != dropped_seed]
        result[str(dropped_seed)] = (sum(remaining) / len(remaining)) - overall
    return result


def summarize_by_point_type(points: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        grouped[str(point.get("point_type", "unknown"))].append(point)
    summary: dict[str, dict[str, float | int]] = {}
    for point_type, rows in sorted(grouped.items()):
        lifts = [point_lift(row) for row in rows]
        summary[point_type] = {
            "count": len(rows),
            "accepted_count": sum(1 for row in rows if row.get("accepted")),
            "point_pooled_lift": sum(lifts) / len(lifts) if lifts else 0.0,
        }
    return summary


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

    matched_lift_values = [point_lift(point) for point in all_points]
    point_pooled_lift = (
        sum(matched_lift_values) / len(matched_lift_values) if matched_lift_values else 0.0
    )

    seed_lifts = seed_level_lifts(all_points)
    seed_level_mean_lift = sum(seed_lifts.values()) / len(seed_lifts) if seed_lifts else 0.0
    ci_low, ci_high = bootstrap_ci(seed_lifts)

    candidate_positive_seeds = {
        int(point["env_seed"])
        for point in all_points
        if point.get("point_type") != "random_negative_control"
        and float(point.get("candidate_success_rate", 0)) > 0
    }
    candidate_positive_rate_by_seed = (
        len(candidate_positive_seeds) / len(seeds) if seeds else 0.0
    )

    accepted_recovery_seeds = {
        int(point["env_seed"])
        for point in accepted
        if point.get("point_type") != "random_negative_control" and point_lift(point) > 0
    }
    accepted_recovery_seed_fraction = (
        len(accepted_recovery_seeds) / len(seeds) if seeds else 0.0
    )

    max_seed_share = max(accepted_by_seed.values()) / len(accepted) if accepted else 0.0

    checks = {
        "harness_valid": bool(pilot.get("harness_valid")) and bool(confirm.get("harness_valid")),
        "lift_positive": seed_level_mean_lift > 0,
        "lift_or_recovery_gate": seed_level_mean_lift >= 0.20 or candidate_positive_rate_by_seed >= 0.20,
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
        "matched_success_lift": point_pooled_lift,
        "point_pooled_lift": point_pooled_lift,
        "seed_level_mean_lift": seed_level_mean_lift,
        "seed_level_bootstrap_ci": {"low": ci_low, "high": ci_high},
        "seed_level_lifts": {str(seed): lift for seed, lift in seed_lifts.items()},
        "leave_one_seed_out_delta": leave_one_seed_out(seed_lifts),
        "by_point_type": summarize_by_point_type(all_points),
        "candidate_positive_rate_by_seed": candidate_positive_rate_by_seed,
        "accepted_recovery_seed_fraction": accepted_recovery_seed_fraction,
        "recovery_seed_fraction": candidate_positive_rate_by_seed,
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
        "schema_version": 2,
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
