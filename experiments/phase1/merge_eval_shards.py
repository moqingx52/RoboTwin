#!/usr/bin/env python3
import argparse
from pathlib import Path

from common import add_common_args, read_json, repo_path, write_json


def summarize(rows, split_name):
    split_rows = [row for row in rows if row["split"] == split_name]
    if not split_rows:
        return {}
    by_seed = {}
    for row in split_rows:
        by_seed.setdefault(row["env_seed"], []).append(row["success"])
    solved = sum(1 for vals in by_seed.values() if any(vals))
    return {
        "episodes": len(split_rows),
        "seeds": len(by_seed),
        "mean_sr": sum(row["success"] for row in split_rows) / len(split_rows),
        "solved_coverage": solved / len(by_seed),
    }


def main():
    parser = argparse.ArgumentParser(description="Merge sharded phase1 eval JSON into one result file.")
    add_common_args(parser)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    task_dir = args.output_dir / args.task_name
    shard_paths = sorted(task_dir.glob(f"{args.variant}_shard_*_of_{args.num_shards:02d}.json"))
    if len(shard_paths) != args.num_shards:
        raise RuntimeError(
            f"Expected {args.num_shards} shards for {args.task_name}/{args.variant}, "
            f"found {len(shard_paths)} under {task_dir}"
        )

    rows = []
    meta = None
    seen = set()
    for path in shard_paths:
        payload = read_json(path)
        if meta is None:
            meta = payload
        for row in payload["rows"]:
            key = (row["split"], int(row["env_seed"]), int(row["repeat"]))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    rows.sort(key=lambda row: (row["split"], int(row["env_seed"]), int(row["repeat"])))

    summary = {
        "task_name": meta["task_name"],
        "task_config": meta["task_config"],
        "variant": meta["variant"],
        "ckpt_path": meta["ckpt_path"],
        "splits": {
            "id_heldout": summarize(rows, "id_heldout"),
            "train_seen": summarize(rows, "train_seen"),
            "hard_20": summarize(rows, "hard_20"),
        },
        "hard_seeds": meta["hard_seeds"],
        "hard_seed_source": meta["hard_seed_source"],
        "rows": rows,
        "progress": {
            "complete": True,
            "completed_episodes": len(rows),
            "id_repeats": meta.get("progress", {}).get("id_repeats"),
            "train_repeats": meta.get("progress", {}).get("train_repeats"),
            "hard_repeats": meta.get("progress", {}).get("hard_repeats"),
            "policy_seed_offset": meta.get("progress", {}).get("policy_seed_offset"),
            "shard_id": 0,
            "num_shards": 1,
            "merged_from_shards": args.num_shards,
        },
    }
    out_path = task_dir / f"{args.variant}.json"
    write_json(out_path, summary)
    print(f"Merged {len(shard_paths)} shards into {out_path}")


if __name__ == "__main__":
    main()
