#!/usr/bin/env python3
import argparse
from pathlib import Path

from common import add_common_args, iter_jsonl, read_json, repo_path, write_json


def main():
    parser = argparse.ArgumentParser(description="Merge phase1 rollout shard manifests into canonical files.")
    add_common_args(parser)
    parser.add_argument("--seeds-file", type=Path)
    parser.add_argument("--rollouts-per-seed", type=int, default=8)
    parser.add_argument("--rollout-dir", type=Path, default=repo_path("experiments", "phase1", "rollouts"))
    args = parser.parse_args()

    seeds_file = args.seeds_file or repo_path("experiments", "phase1", "seeds", f"{args.task_name}_seeds.json")
    seeds_payload = read_json(seeds_file)
    train_seeds = [int(seed) for seed in seeds_payload["train_rollout"]]
    task_dir = args.rollout_dir / args.task_name
    shard_paths = sorted(task_dir.glob("manifest_shard_*_of_*.jsonl"))
    if not shard_paths:
        shard_paths = [task_dir / "manifest.jsonl"]

    rows = []
    seen = set()
    for path in shard_paths:
        for row in iter_jsonl(path):
            key = (int(row["env_seed"]), int(row["rollout_id"]))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    rows.sort(key=lambda row: (int(row["env_seed"]), int(row["rollout_id"])))

    manifest_path = task_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        import json
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    by_seed = {}
    for seed in train_seeds:
        seed_rows = [row for row in rows if int(row["env_seed"]) == seed]
        success_count = sum(1 for row in seed_rows if row["success"])
        by_seed[str(seed)] = {
            "attempts": len(seed_rows),
            "successes": success_count,
            "j_hat": success_count / args.rollouts_per_seed if args.rollouts_per_seed else 0.0,
        }
    stats_path = task_dir / "seed_stats.json"
    write_json(stats_path, by_seed)
    print(f"Merged {len(shard_paths)} shard manifests into {manifest_path}")
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()

