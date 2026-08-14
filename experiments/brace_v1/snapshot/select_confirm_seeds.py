#!/usr/bin/env python3
"""Select held-out mixed-outcome seeds for confirmatory Stage-2 runs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for import_path in (REPO_ROOT, REPO_ROOT / "experiments" / "phase1"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from experiments.brace.replay_audit import iter_manifest_rows, read_json, repo_path, write_json_atomic
from experiments.brace.select_pilot_seeds import select_mixed_outcome_seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--exclude-seeds", nargs="*", type=int, default=[])
    parser.add_argument("--exclude-seeds-file", type=Path, default=None)
    parser.add_argument("--rollout-id-min", type=int, default=None)
    parser.add_argument("--rollout-id-max", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    exclude = set(args.exclude_seeds)
    if args.exclude_seeds_file is not None:
        payload = read_json(repo_path(args.exclude_seeds_file))
        exclude.update(int(seed) for seed in payload.get("seeds", []))
        exclude.update(int(seed) for seed in payload.get("analyzed_seeds", []))

    task_dir = repo_path(args.rollout_dir) / args.task
    rows = list(iter_manifest_rows(task_dir))
    if args.rollout_id_min is not None or args.rollout_id_max is not None:
        rollout_min = args.rollout_id_min if args.rollout_id_min is not None else 0
        rollout_max = args.rollout_id_max if args.rollout_id_max is not None else 10**9
        rows = [row for row in rows if rollout_min <= int(row["rollout_id"]) <= rollout_max]

    mixed = [seed for seed in select_mixed_outcome_seeds(rows, count=10**9, seed=args.seed) if seed not in exclude]
    rng = random.Random(args.seed)
    rng.shuffle(mixed)
    selected = mixed[: args.count]
    if len(selected) < args.count:
        raise SystemExit(
            f"only found {len(selected)} held-out mixed-outcome seeds for {args.task} "
            f"(exclude={sorted(exclude)}, need {args.count})"
        )

    output_payload = {
        "task": args.task,
        "run_type": "confirmatory",
        "selection_seed": args.seed,
        "requested_count": args.count,
        "seeds": selected,
        "excluded_seeds": sorted(exclude),
        "rollout_id_range": [args.rollout_id_min, args.rollout_id_max],
        "source_rollout_dir": str(repo_path(args.rollout_dir)),
    }
    output = repo_path(args.output)
    write_json_atomic(output, output_payload)
    try:
        from experiments.brace.stage_records import emit_stage_record

        emit_stage_record(
            "select_confirm_seeds",
            summary=output_payload,
            summary_path=output,
            tasks=[args.task],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to emit stage record: {exc}", file=sys.stderr)
    print(json.dumps(output_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
