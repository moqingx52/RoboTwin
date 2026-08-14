#!/usr/bin/env python3
"""Inventory traced rollouts for directed pilot collection (avoid unbounded audit)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for import_path in (REPO_ROOT, REPO_ROOT / "experiments" / "phase1"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from experiments.brace.replay_audit import iter_manifest_rows, repo_path


def inventory_task(task_dir: Path) -> dict:
    by_seed: dict[int, dict[str, int]] = defaultdict(lambda: {"success": 0, "failure": 0})
    for row in iter_manifest_rows(task_dir):
        env_seed = int(row["env_seed"])
        if row.get("success"):
            by_seed[env_seed]["success"] += 1
        else:
            by_seed[env_seed]["failure"] += 1

    mixed = sorted(seed for seed, stats in by_seed.items() if stats["success"] > 0 and stats["failure"] > 0)
    return {
        "total_rows": sum(stats["success"] + stats["failure"] for stats in by_seed.values()),
        "env_seeds": len(by_seed),
        "mixed_outcome_seeds": mixed,
        "mixed_outcome_count": len(mixed),
        "per_seed": {str(seed): by_seed[seed] for seed in sorted(by_seed)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    task_dir = repo_path(args.rollout_dir) / args.task
    if not task_dir.is_dir():
        raise SystemExit(f"missing task directory: {task_dir}")

    payload = {"task": args.task, "rollout_dir": str(repo_path(args.rollout_dir)), **inventory_task(task_dir)}
    text = json.dumps(payload, indent=2)
    if args.output:
        out = repo_path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
