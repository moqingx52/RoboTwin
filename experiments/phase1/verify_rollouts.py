#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

from common import TASKS, repo_path


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Strictly verify completed phase1 rollout shards.")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--rollouts-per-seed", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=2)
    args = parser.parse_args()

    import h5py

    failed = False
    for task in args.tasks:
        seed_path = repo_path("experiments", "phase1", "seeds", f"{task}_seeds.json")
        with seed_path.open(encoding="utf-8") as f:
            seed_payload = json.load(f)
        seeds = [int(seed) for seed in seed_payload["train_rollout"]]
        expected = {(seed, rollout) for seed in seeds for rollout in range(args.rollouts_per_seed)}

        task_dir = args.rollout_dir / task
        shards = sorted(task_dir.glob(f"manifest_shard_*_of_{args.num_shards:02d}.jsonl"))
        rows = [row for shard in shards for row in read_jsonl(shard)]
        keys = [(int(row["env_seed"]), int(row["rollout_id"])) for row in rows]
        counts = Counter(keys)
        missing = expected - set(keys)
        extra = set(keys) - expected
        duplicates = sum(count - 1 for count in counts.values() if count > 1)

        successes = [row for row in rows if row.get("success")]
        invalid_paths = []
        inconsistent_rows = 0
        for row in rows:
            has_path = row.get("hdf5_path") is not None
            if has_path != bool(row.get("success")):
                inconsistent_rows += 1
            if not row.get("success"):
                continue
            path = Path(row["hdf5_path"])
            if not path.is_absolute():
                path = repo_path(path)
            try:
                with h5py.File(path, "r") as root:
                    if root["/joint_action/vector"].shape[0] <= 1:
                        raise ValueError("empty joint trajectory")
                    if root["/observation/head_camera/rgb"].shape[0] <= 1:
                        raise ValueError("empty head-camera trajectory")
            except Exception as exc:
                invalid_paths.append(f"{path}: {exc}")

        ok = (
            len(shards) == args.num_shards
            and not missing
            and not extra
            and duplicates == 0
            and not invalid_paths
            and inconsistent_rows == 0
        )
        failed |= not ok
        rate = len(successes) / len(rows) if rows else 0.0
        print(
            f"[{'PASS' if ok else 'FAIL'}] {task}: shards={len(shards)}, "
            f"rows={len(rows)}/{len(expected)}, success={len(successes)} ({rate:.3f}), "
            f"missing={len(missing)}, extra={len(extra)}, duplicates={duplicates}, "
            f"bad_hdf5={len(invalid_paths)}, inconsistent={inconsistent_rows}"
        )
        if seed_payload.get("meta", {}).get("dry_run"):
            print("  WARNING: seed split was generated with --dry-run")
            failed = True
        for message in invalid_paths[:3]:
            print(f"  {message}")

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
