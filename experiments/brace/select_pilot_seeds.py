#!/usr/bin/env python3
"""Select mixed-outcome env seeds for BRACE Stage-2 pilot collection."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE1_DIR = REPO_ROOT / "experiments" / "phase1"
for import_path in (REPO_ROOT, PHASE1_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from experiments.brace.replay_audit import iter_manifest_rows, load_env_seeds_from_json, read_json, repo_path, write_json_atomic


def select_mixed_outcome_seeds(
    rows: list[dict],
    *,
    count: int,
    seed: int,
) -> list[int]:
    by_seed: dict[int, dict[str, int]] = defaultdict(lambda: {"success": 0, "failure": 0})
    for row in rows:
        env_seed = int(row["env_seed"])
        if row.get("success"):
            by_seed[env_seed]["success"] += 1
        else:
            by_seed[env_seed]["failure"] += 1
    mixed = sorted(
        env_seed
        for env_seed, stats in by_seed.items()
        if stats["success"] > 0 and stats["failure"] > 0
    )
    rng = random.Random(seed)
    rng.shuffle(mixed)
    return mixed[:count]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    task_dir = repo_path(args.rollout_dir) / args.task
    rows = list(iter_manifest_rows(task_dir))
    seeds = select_mixed_outcome_seeds(rows, count=args.count, seed=args.seed)
    if len(seeds) < args.count:
        raise SystemExit(
            f"only found {len(seeds)} mixed-outcome seeds for {args.task}, need {args.count}"
        )
    payload = {
        "task": args.task,
        "selection_seed": args.seed,
        "requested_count": args.count,
        "seeds": seeds,
        "source_rollout_dir": str(repo_path(args.rollout_dir)),
    }
    output = repo_path(args.output)
    write_json_atomic(output, payload)
    try:
        from experiments.brace.stage_records import emit_stage_record

        emit_stage_record(
            "select_pilot_seeds",
            summary=payload,
            summary_path=output,
            tasks=[args.task],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to emit stage record: {exc}", file=sys.stderr)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
