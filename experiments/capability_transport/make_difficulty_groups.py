#!/usr/bin/env python3
"""Freeze Beta-Binomial soft difficulty groups from a completed T1b 240x8 run.

Reads the merged eval_per_seed result (t1_difficulty_v1.json), fits the
per-seed posterior p0 | s ~ Beta(1 + s, 1 + R_eval - s) over evaluated
repeats only, assigns soft groups by posterior mean, and writes
difficulty_groups.v1.json (+ .sha256) next to the protocol.

Freeze rule (protocol.t1.v1.json): run once on the pi0 baseline result and
never again; post-treatment checkpoints must never feed this script.
"""

import argparse
import hashlib
import json
from pathlib import Path

from common import read_json, write_json_atomic

PROTOCOL_REVISION = "capability_transport.t1.v1"
EXPECTED_REPEATS = 8
MIN_EVALUATED_REPEATS = 6
GROUP_BOUNDS = {"easy": (0.7, 1.0 + 1e-9), "medium": (0.3, 0.7), "hard": (0.0, 0.3)}


def posterior_mean(successes: int, evaluated: int) -> float:
    return (1 + successes) / (2 + evaluated)


def assign_group(mean: float) -> str:
    for name, (lo, hi) in GROUP_BOUNDS.items():
        if lo <= mean < hi:
            return name
    raise ValueError(f"posterior mean {mean} outside [0, 1]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True,
                        help="Merged t1_difficulty_v1.json from run_t1_difficulty.sh")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path for difficulty_groups.v1.json")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing frozen output (audit trail: do not use after freeze)")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite frozen groups: {args.output}")

    payload = read_json(args.result)
    if not payload.get("progress", {}).get("complete"):
        raise SystemExit(f"Result is not complete: {args.result}")

    per_seed = {}
    for row in payload["rows"]:
        entry = per_seed.setdefault(int(row["env_seed"]), {"split": row["split"], "successes": 0, "evaluated": 0, "missing": 0})
        if row.get("evaluated", True):
            entry["evaluated"] += 1
            entry["successes"] += int(bool(row["success"]))
        else:
            entry["missing"] += 1

    groups = {"easy": [], "medium": [], "hard": []}
    unsupported_tail = []
    excluded_insufficient_repeats = []
    seed_records = {}
    for seed in sorted(per_seed):
        entry = per_seed[seed]
        total = entry["evaluated"] + entry["missing"]
        if total != EXPECTED_REPEATS:
            raise SystemExit(f"Seed {seed} has {total} rows, expected {EXPECTED_REPEATS}")
        record = {
            "split": entry["split"],
            "successes": entry["successes"],
            "evaluated_repeats": entry["evaluated"],
            "operational_missing_repeats": entry["missing"],
        }
        if entry["evaluated"] < MIN_EVALUATED_REPEATS:
            record["group"] = None
            record["excluded_reason"] = f"evaluated_repeats < {MIN_EVALUATED_REPEATS}"
            excluded_insufficient_repeats.append(seed)
        else:
            mean = posterior_mean(entry["successes"], entry["evaluated"])
            group = assign_group(mean)
            record["posterior_mean"] = round(mean, 6)
            record["group"] = group
            groups[group].append(seed)
            if entry["successes"] == 0 and entry["evaluated"] == EXPECTED_REPEATS:
                unsupported_tail.append(seed)
        seed_records[str(seed)] = record

    result_sha = hashlib.sha256(args.result.read_bytes()).hexdigest()
    out = {
        "schema_version": 1,
        "protocol_revision": PROTOCOL_REVISION,
        "task": payload["task_name"],
        "ckpt_path": payload["ckpt_path"],
        "source_result": str(args.result),
        "source_result_sha256": result_sha,
        "policy_seed_offset": payload["progress"]["policy_seed_offset"],
        "expected_repeats": EXPECTED_REPEATS,
        "min_evaluated_repeats": MIN_EVALUATED_REPEATS,
        "posterior": "p0 | s ~ Beta(1 + s, 1 + R_eval - s); grouped by posterior mean",
        "group_bounds": {"easy": "[0.7, 1.0]", "medium": "[0.3, 0.7)", "hard": "[0.0, 0.3)"},
        "groups": {name: sorted(seeds) for name, seeds in groups.items()},
        "group_sizes": {name: len(seeds) for name, seeds in groups.items()},
        "unsupported_tail_0_of_8": sorted(unsupported_tail),
        "excluded_insufficient_repeats": sorted(excluded_insufficient_repeats),
        "seeds": seed_records,
        "freeze_note": "Frozen from the independent pi0 baseline only; never regroup after training.",
    }
    write_json_atomic(args.output, out)
    out_sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(f"{out_sha}  {args.output.name}\n")
    print(f"Wrote {args.output} ({out_sha})")
    print("Group sizes:", out["group_sizes"], "| unsupported tail:", len(unsupported_tail),
          "| excluded:", len(excluded_insufficient_repeats))


if __name__ == "__main__":
    main()
