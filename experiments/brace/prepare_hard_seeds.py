#!/usr/bin/env python3
"""Prepare held-out hard eval seeds from a minimal base ID probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE1_DIR = REPO_ROOT / "experiments" / "phase1"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.replay_audit import git_commit, read_json, repo_path, write_json_atomic


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def base_checkpoint(task: str) -> Path:
    return REPO_ROOT / "policy" / "DP" / "checkpoints" / f"{task}-demo_clean-200-0" / "600.ckpt"


def merged_probe_path(probe_dir: Path, task: str) -> Path:
    return probe_dir / task / "base.json"


def run_probe_shard(
    *,
    task: str,
    ckpt_path: Path,
    probe_dir: Path,
    shard_id: int,
    num_shards: int,
    resume: bool,
) -> None:
    cmd = [
        sys.executable,
        str(PHASE1_DIR / "eval_per_seed.py"),
        "--task",
        task,
        "--task-config",
        "demo_clean",
        "--variant",
        "base",
        "--ckpt-path",
        str(ckpt_path),
        "--rollout-dir",
        str(REPO_ROOT / "experiments" / "phase1" / "rollouts_200"),
        "--output-dir",
        str(probe_dir),
        "--id-repeats",
        "8",
        "--train-repeats",
        "0",
        "--hard-repeats",
        "0",
        "--no-include-hard",
        "--policy-seed-offset",
        "0",
        "--shard-id",
        str(shard_id),
        "--num-shards",
        str(num_shards),
    ]
    if resume:
        cmd.append("--resume")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def merge_probe(task: str, probe_dir: Path, num_shards: int) -> Path:
    cmd = [
        sys.executable,
        str(PHASE1_DIR / "merge_eval_shards.py"),
        "--task",
        task,
        "--task-config",
        "demo_clean",
        "--variant",
        "base",
        "--output-dir",
        str(probe_dir),
        "--num-shards",
        str(num_shards),
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    merged = merged_probe_path(probe_dir, task)
    if not merged.is_file():
        raise FileNotFoundError(f"merged base probe missing: {merged}")
    return merged


def select_hard_seeds(task: str, base_eval: Path, hard_output: Path, count: int) -> list[int]:
    hard_output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(PHASE1_DIR / "select_hard_eval_seeds.py"),
        "--base-eval",
        str(base_eval),
        "--count",
        str(count),
        "--output",
        str(hard_output),
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    payload = read_json(hard_output)
    return [int(seed) for seed in payload["hard_seeds"]]


def probe_complete(task: str, probe_dir: Path, num_shards: int) -> bool:
    merged = merged_probe_path(probe_dir, task)
    if merged.is_file():
        return True
    cmd = [
        sys.executable,
        str(PHASE1_DIR / "eval_per_seed.py"),
        "--task",
        task,
        "--task-config",
        "demo_clean",
        "--variant",
        "base",
        "--ckpt-path",
        str(base_checkpoint(task)),
        "--output-dir",
        str(probe_dir),
        "--shard-id",
        "0",
        "--num-shards",
        str(num_shards),
        "--check-complete-result",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    return result.returncode == 0


def build_summary(
    *,
    task: str,
    probe_dir: Path,
    hard_output: Path,
    hard_seeds: list[int],
    num_shards: int,
    probe_complete_flag: bool,
) -> dict:
    base_eval = merged_probe_path(probe_dir, task)
    return {
        "schema_version": 1,
        "stage": "prepare_hard_seeds",
        "task": task,
        "passed": probe_complete_flag and hard_output.is_file(),
        "complete": probe_complete_flag and hard_output.is_file(),
        "num_shards": num_shards,
        "probe_complete": probe_complete_flag,
        "base_probe_path": str(base_eval),
        "base_probe_sha256": sha256_file(base_eval) if base_eval.is_file() else None,
        "hard_seeds_path": str(hard_output),
        "hard_seeds": hard_seeds,
        "hard_seeds_sha256": sha256_file(hard_output) if hard_output.is_file() else None,
        "git_commit": git_commit(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--ckpt-path", type=Path, default=None)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--hard-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    ckpt_path = repo_path(args.ckpt_path) if args.ckpt_path else base_checkpoint(args.task)
    probe_dir = repo_path(args.probe_dir)
    hard_output = repo_path(args.hard_output)
    if not ckpt_path.is_file():
        raise SystemExit(f"missing checkpoint: {ckpt_path}")

    if args.shard_id is not None:
        run_probe_shard(
            task=args.task,
            ckpt_path=ckpt_path,
            probe_dir=probe_dir,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            resume=args.resume,
        )
        return 0

    if not args.merge_only:
        for shard_id in range(args.num_shards):
            run_probe_shard(
                task=args.task,
                ckpt_path=ckpt_path,
                probe_dir=probe_dir,
                shard_id=shard_id,
                num_shards=args.num_shards,
                resume=args.resume,
            )

    merged = merge_probe(args.task, probe_dir, args.num_shards)
    hard_seeds = select_hard_seeds(args.task, merged, hard_output, args.count)
    complete = probe_complete(args.task, probe_dir, args.num_shards)
    summary = build_summary(
        task=args.task,
        probe_dir=probe_dir,
        hard_output=hard_output,
        hard_seeds=hard_seeds,
        num_shards=args.num_shards,
        probe_complete_flag=complete,
    )
    if args.summary_output is not None:
        summary_path = repo_path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(summary_path, summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
