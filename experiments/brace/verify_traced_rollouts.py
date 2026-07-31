#!/usr/bin/env python3
"""Verify BRACE traced rollout shards include schema v2 extensions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE1_DIR = REPO_ROOT / "experiments" / "phase1"
for import_path in (REPO_ROOT, PHASE1_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from common import TASKS, repo_path
from experiments.brace.control_trace import validate_schema_v2


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _row_label(row: dict) -> str:
    return f"seed={row['env_seed']} rollout={row['rollout_id']}"


def _row_hdf5_path(row: dict) -> Path | None:
    path_value = row.get("hdf5_path") if row.get("success") else row.get("failure_hdf5_path")
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = repo_path(path)
    return path


def _validate_row(job: tuple[str, str]) -> list[str]:
    label, path_text = job
    schema_errors = validate_schema_v2(Path(path_text))
    return [f"{label}: {error}" for error in schema_errors]


def validate_rows(rows: list[dict], workers: int) -> list[str]:
    jobs: list[tuple[str, str]] = []
    invalid_paths: list[str] = []
    for row in rows:
        label = _row_label(row)
        path = _row_hdf5_path(row)
        if path is None:
            invalid_paths.append(f"{label}: missing path")
            continue
        jobs.append((label, str(path)))

    if workers <= 1 or len(jobs) <= 1:
        for label, path_text in jobs:
            invalid_paths.extend(_validate_row((label, path_text)))
        return invalid_paths

    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_validate_row, job): job for job in jobs}
        for future in as_completed(futures):
            invalid_paths.extend(future.result())
            completed += 1
            if completed == 1 or completed % 50 == 0 or completed == len(jobs):
                print(f"  validated {completed}/{len(jobs)} HDF5 files", flush=True)
    return invalid_paths


def main():
    parser = argparse.ArgumentParser(description="Verify traced rollout HDF5 schema v2.")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--rollouts-per-seed", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--require-failures", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel HDF5 schema validators (manifest checks stay serial).",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    failed = False
    for task in args.tasks:
        seed_path = repo_path("experiments", "phase1", "seeds", f"{task}_seeds.json")
        with seed_path.open(encoding="utf-8") as handle:
            seed_payload = json.load(handle)
        seeds = [int(seed) for seed in seed_payload["train_rollout"]]
        expected = {(seed, rollout) for seed in seeds for rollout in range(args.rollouts_per_seed)}

        task_dir = args.rollout_dir / task
        shards = sorted(task_dir.glob(f"manifest_shard_*_of_{args.num_shards:02d}.jsonl"))
        if not shards and (task_dir / "manifest.jsonl").is_file():
            shards = [task_dir / "manifest.jsonl"]
        rows = [row for shard in shards for row in read_jsonl(shard)]
        keys = [(int(row["env_seed"]), int(row["rollout_id"])) for row in rows]
        counts = Counter(keys)
        missing = expected - set(keys)
        extra = set(keys) - expected
        duplicates = sum(count - 1 for count in counts.values() if count > 1)

        print(f"Validating {task}: {len(rows)} manifest rows with workers={args.workers}", flush=True)
        invalid_paths = validate_rows(rows, args.workers)

        ok = not missing and not extra and duplicates == 0 and not invalid_paths
        failed |= not ok
        print(
            f"[{'PASS' if ok else 'FAIL'}] {task}: rows={len(rows)}/{len(expected)}, "
            f"missing={len(missing)}, extra={len(extra)}, duplicates={duplicates}, "
            f"bad_trace={len(invalid_paths)}"
        )
        for message in invalid_paths[:3]:
            print(f"  {message}")

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
