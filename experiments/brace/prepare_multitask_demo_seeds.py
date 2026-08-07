#!/usr/bin/env python3
"""Write rollout_train seeds (first N) into data/<task>/demo_clean/seed.txt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=BRACE_DIR / "seeds" / "multitask_v1",
    )
    parser.add_argument("--partition", default="rollout_train")
    args = parser.parse_args()

    manifest_path = args.manifest / f"{args.task}.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    seeds = payload["partitions"][args.partition][: args.count]
    if len(seeds) < args.count:
        raise SystemExit(f"{args.task}: only {len(seeds)} seeds in {args.partition}, need {args.count}")

    save_dir = REPO_ROOT / "data" / args.task / "demo_clean"
    save_dir.mkdir(parents=True, exist_ok=True)
    seed_path = save_dir / "seed.txt"
    seed_path.write_text(" ".join(str(seed) for seed in seeds) + "\n", encoding="utf-8")
    print(f"wrote {len(seeds)} seeds to {seed_path} ({seeds[0]}..{seeds[-1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
