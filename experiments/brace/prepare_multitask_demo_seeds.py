#!/usr/bin/env python3
"""Write frozen expert_demo seeds into data/<task>/demo_clean/seed.txt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.seed_feasibility import DEFAULT_EXPERT_DEMO_COUNT


def resolve_expert_demo_seeds(manifest: dict, *, count: int) -> list[int]:
    selection = manifest.get("expert_demo_selection", {})
    seeds = manifest.get("partitions", {}).get("expert_demo", [])
    if not seeds:
        raise SystemExit(
            f"{manifest.get('task', '<task>')}: partitions.expert_demo is empty; "
            "run scan_seed_feasibility.py --update-manifest first"
        )
    if len(seeds) != count:
        raise SystemExit(
            f"expected {count} expert_demo seeds, found {len(seeds)} "
            f"(rule={selection.get('rule')})"
        )
    rollout_train = set(manifest.get("partitions", {}).get("rollout_train", []))
    if not set(seeds).issubset(rollout_train):
        raise SystemExit("expert_demo seeds must remain a subset of rollout_train")
    return [int(seed) for seed in seeds]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--count", type=int, default=DEFAULT_EXPERT_DEMO_COUNT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=BRACE_DIR / "seeds" / "multitask_v1",
    )
    args = parser.parse_args()

    manifest_path = args.manifest / f"{args.task}.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    seeds = resolve_expert_demo_seeds(payload, count=args.count)

    save_dir = REPO_ROOT / "data" / args.task / "demo_clean"
    save_dir.mkdir(parents=True, exist_ok=True)
    seed_path = save_dir / "seed.txt"
    seed_path.write_text(" ".join(str(seed) for seed in seeds) + "\n", encoding="utf-8")
    print(
        f"wrote {len(seeds)} expert_demo seeds to {seed_path} "
        f"({seeds[0]}..{seeds[-1]}) rule={payload.get('expert_demo_selection', {}).get('rule')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
