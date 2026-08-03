#!/usr/bin/env python3
"""Run one logical screen evaluation as concurrent shards on one assigned GPU."""

from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE1_DIR = REPO_ROOT / "experiments" / "phase1"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))

from common import read_json, write_json_atomic  # noqa: E402
from eval_per_seed import build_work_items, summarize, work_item_key  # noqa: E402


def option(command: list[str], name: str, default=None):
    try:
        return command[command.index(name) + 1]
    except ValueError:
        return default


def output_path(command: list[str]) -> Path:
    output_dir = Path(option(command, "--output-dir"))
    task = option(command, "--task")
    variant = option(command, "--variant")
    return output_dir / task / f"{variant}.json"


def shard_path(command: list[str], shard: int, workers: int) -> Path:
    final = output_path(command)
    return final.with_name(f"{final.stem}_shard_{shard:02d}_of_{workers:02d}.json")


def load_extra_splits(command: list[str]) -> dict[str, list[int]] | None:
    path = option(command, "--extra-splits-file")
    if not path:
        return None
    payload = read_json(Path(path))
    return {str(key): [int(seed) for seed in value] for key, value in payload.items()}


def expected_episode_count(command: list[str]) -> int:
    seed_payload, hard = load_seed_payload(command)
    extra_splits = load_extra_splits(command)
    id_repeats = int(option(command, "--id-repeats", 1))
    train_repeats = int(option(command, "--train-repeats", 1))
    hard_repeats = int(option(command, "--hard-repeats", 1))
    extra_split_repeats = int(option(command, "--extra-split-repeats", 1))
    items = build_work_items(
        seed_payload, hard, id_repeats, train_repeats, hard_repeats, extra_splits, extra_split_repeats
    )
    return len(items)


def load_seed_payload(command: list[str]) -> tuple[dict, list[int]]:
    task = option(command, "--task")
    seeds_file = option(command, "--seeds-file")
    if seeds_file is None:
        seeds_file = PHASE1_DIR / "seeds" / f"{task}_seeds.json"
    payload = read_json(Path(seeds_file))
    payload = dict(payload)
    id_count = option(command, "--id-seed-count")
    train_count = option(command, "--train-seed-count")
    if id_count is not None:
        payload["eval_id"] = list(payload["eval_id"])[: int(id_count)]
    if train_count is not None:
        payload["train_rollout"] = list(payload["train_rollout"])[: int(train_count)]

    hard_file = option(command, "--hard-seeds-file")
    if hard_file is None:
        hard = list(payload["train_rollout"])[:20]
    else:
        hard_payload = read_json(Path(hard_file))
        hard = hard_payload["hard_seeds"] if isinstance(hard_payload, dict) else hard_payload
    hard_count = option(command, "--hard-seed-count")
    if hard_count is not None:
        hard = list(hard)[: int(hard_count)]
    if "--no-include-hard" in command:
        hard = []
    return payload, [int(seed) for seed in hard]


def seed_shards_from_partial(command: list[str], workers: int) -> int:
    """Redistribute compatible unsharded resume rows into shard result files."""
    final = output_path(command)
    if not final.is_file():
        return 0
    parent = read_json(final)
    progress = parent.get("progress", {})
    if progress.get("complete"):
        return 0
    rows = parent.get("rows", [])
    if not rows:
        return 0

    seed_payload, hard = load_seed_payload(command)
    extra_splits = load_extra_splits(command)
    id_repeats = int(option(command, "--id-repeats", 3))
    train_repeats = int(option(command, "--train-repeats", 3))
    hard_repeats = int(option(command, "--hard-repeats", 8))
    extra_split_repeats = int(option(command, "--extra-split-repeats", 1))
    items = build_work_items(
        seed_payload, hard, id_repeats, train_repeats, hard_repeats, extra_splits, extra_split_repeats
    )
    expected_meta = {
        "task_name": option(command, "--task"),
        "task_config": option(command, "--task-config", "demo_clean"),
        "variant": option(command, "--variant"),
        "ckpt_path": option(command, "--ckpt-path"),
    }
    mismatches = [
        f"{key}={parent.get(key)!r}, expected {value!r}"
        for key, value in expected_meta.items()
        if parent.get(key) != value
    ]
    expected_progress = {
        "id_repeats": id_repeats,
        "train_repeats": train_repeats,
        "hard_repeats": hard_repeats,
        "extra_split_repeats": extra_split_repeats,
        "policy_seed_offset": int(option(command, "--policy-seed-offset", 0)),
    }
    mismatches.extend(
        f"progress.{key}={progress.get(key)!r}, expected {value!r}"
        for key, value in expected_progress.items()
        if key in progress and progress.get(key) != value
    )
    if mismatches:
        raise RuntimeError(f"Refusing to migrate incompatible partial result {final}: " + "; ".join(mismatches))
    item_shard = {work_item_key(*item): index % workers for index, item in enumerate(items)}
    distributed = [[] for _ in range(workers)]
    for row in rows:
        key = work_item_key(row["split"], row["env_seed"], row["repeat"])
        if key not in item_shard:
            raise RuntimeError(f"Partial result {final} has unexpected work item {key}")
        expected_policy_seed = expected_progress["policy_seed_offset"] + int(row["repeat"])
        if int(row.get("policy_seed", -1)) != expected_policy_seed:
            raise RuntimeError(
                f"Partial result {final} has policy_seed={row.get('policy_seed')!r} for {key}, "
                f"expected {expected_policy_seed}"
            )
        distributed[item_shard[key]].append(row)

    for shard, shard_rows in enumerate(distributed):
        path = shard_path(command, shard, workers)
        if path.is_file():
            continue
        expected = sum(1 for index in range(len(items)) if index % workers == shard)
        payload = copy.deepcopy(parent)
        payload["rows"] = shard_rows
        payload["splits"] = {
            split: summarize(shard_rows, split)
            for split in sorted({row["split"] for row in shard_rows} or {"id_heldout", "train_seen", "hard_20"})
        }
        payload["progress"] = {
            "complete": len(shard_rows) == expected,
            "completed_episodes": len(shard_rows),
            "id_repeats": id_repeats,
            "train_repeats": train_repeats,
            "hard_repeats": hard_repeats,
            "extra_split_repeats": extra_split_repeats,
            "policy_seed_offset": int(option(command, "--policy-seed-offset", 0)),
            "shard_id": shard,
            "num_shards": workers,
        }
        write_json_atomic(path, payload)
    return len(rows)


def with_shard(command: list[str], shard: int, workers: int) -> list[str]:
    result = list(command)
    for name in ("--shard-id", "--num-shards", "--check-complete-result"):
        while name in result:
            index = result.index(name)
            del result[index : index + (1 if name == "--check-complete-result" else 2)]
    result.extend(["--shard-id", str(shard), "--num-shards", str(workers)])
    if "--resume" not in result:
        result.append("--resume")
    return result


def merge_command(command: list[str], workers: int) -> list[str]:
    return [
        "python",
        str(PHASE1_DIR / "merge_eval_shards.py"),
        "--task",
        option(command, "--task"),
        "--task-config",
        option(command, "--task-config", "demo_clean"),
        "--variant",
        option(command, "--variant"),
        "--output-dir",
        option(command, "--output-dir"),
        "--num-shards",
        str(workers),
    ]


def check_complete_command(command: list[str]) -> list[str]:
    result = list(command)
    for name in ("--shard-id", "--num-shards", "--check-complete-result"):
        while name in result:
            index = result.index(name)
            del result[index : index + (1 if name == "--check-complete-result" else 2)]
    result.append("--check-complete-result")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if not command:
        raise SystemExit("missing eval_per_seed command after --")

    final = output_path(command)
    if final.is_file():
        complete_check = subprocess.run(check_complete_command(command), cwd=REPO_ROOT, check=False)
        if complete_check.returncode == 0:
            print(f"Reusing completed logical evaluation: {final}")
            return 0
    migrated = seed_shards_from_partial(command, args.workers)
    if migrated:
        print(f"Migrated {migrated} partial rows from {final} into {args.workers} shards", flush=True)

    processes: list[subprocess.Popen] = []

    def stop(*_):
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    for shard in range(args.workers):
        processes.append(subprocess.Popen(with_shard(command, shard, args.workers), cwd=REPO_ROOT, env=os.environ.copy()))
    failed = False
    for process in processes:
        if process.wait() != 0:
            failed = True
    if failed:
        return 1
    merged = subprocess.run(merge_command(command, args.workers), cwd=REPO_ROOT, check=False)
    if merged.returncode != 0:
        return merged.returncode
    return subprocess.run(check_complete_command(command), cwd=REPO_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
