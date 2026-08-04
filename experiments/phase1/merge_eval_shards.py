#!/usr/bin/env python3
import argparse
from pathlib import Path

from common import add_common_args, read_json, repo_path, write_json_atomic


PROGRESS_KEYS = (
    "id_repeats",
    "train_repeats",
    "hard_repeats",
    "extra_split_repeats",
    "policy_seed_offset",
)
OPTIONAL_PROGRESS_KEYS = (
    "census_candidate_split",
    "census_candidate_id_count",
)


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


def _merged_progress(shard_progress: dict, completed_episodes: int, num_shards: int) -> dict:
    progress = {
        "complete": True,
        "completed_episodes": completed_episodes,
        "id_repeats": shard_progress.get("id_repeats"),
        "train_repeats": shard_progress.get("train_repeats"),
        "hard_repeats": shard_progress.get("hard_repeats"),
        "extra_split_repeats": shard_progress.get("extra_split_repeats"),
        "policy_seed_offset": shard_progress.get("policy_seed_offset"),
        "shard_id": 0,
        "num_shards": 1,
        "merged_from_shards": num_shards,
    }
    if shard_progress.get("census_candidate_split"):
        progress["census_candidate_split"] = True
        if shard_progress.get("census_candidate_id_count") is not None:
            progress["census_candidate_id_count"] = shard_progress["census_candidate_id_count"]
    return progress


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
        else:
            for key in ("task_name", "task_config", "variant", "ckpt_path", "hard_seeds"):
                if payload.get(key) != meta.get(key):
                    raise RuntimeError(
                        f"Incompatible shard metadata for {path}: {key}={payload.get(key)!r}, "
                        f"expected {meta.get(key)!r}"
                    )
            for key in PROGRESS_KEYS + OPTIONAL_PROGRESS_KEYS:
                if payload.get("progress", {}).get(key) != meta.get("progress", {}).get(key):
                    raise RuntimeError(
                        f"Incompatible shard metadata for {path}: progress.{key}="
                        f"{payload.get('progress', {}).get(key)!r}, expected "
                        f"{meta.get('progress', {}).get(key)!r}"
                    )
        progress = payload.get("progress", {})
        if not progress.get("complete"):
            raise RuntimeError(f"Refusing to merge incomplete shard: {path}")
        if int(progress.get("num_shards", -1)) != args.num_shards:
            raise RuntimeError(
                f"Shard {path} records num_shards={progress.get('num_shards')}, expected {args.num_shards}"
            )
        for row in payload["rows"]:
            key = (row["split"], int(row["env_seed"]), int(row["repeat"]))
            if key in seen:
                raise RuntimeError(f"Duplicate work item across shards: {key}")
            seen.add(key)
            rows.append(row)
    rows.sort(key=lambda row: (row["split"], int(row["env_seed"]), int(row["repeat"])))

    summary = {
        "task_name": meta["task_name"],
        "task_config": meta["task_config"],
        "variant": meta["variant"],
        "ckpt_path": meta["ckpt_path"],
        "splits": {
            split: summarize(rows, split)
            for split in sorted({row["split"] for row in rows})
        },
        "hard_seeds": meta["hard_seeds"],
        "hard_seed_source": meta["hard_seed_source"],
        "rows": rows,
        "progress": _merged_progress(meta.get("progress", {}), len(rows), args.num_shards),
    }
    out_path = task_dir / f"{args.variant}.json"
    write_json_atomic(out_path, summary)
    print(f"Merged {len(shard_paths)} shards into {out_path}")


if __name__ == "__main__":
    main()
