#!/usr/bin/env python3
"""Launch T2 post-train eval with full GPU utilization.

Scheduling model:
- One logical eval job per GPU.
- Each logical eval job runs `run_eval_group.py --workers 3`.
- 8 GPUs -> 8 concurrent logical eval jobs.

Episode policy (v2, slim):
- Roll out unique panels only: easy / medium / hard / memorization_hard.
- Derive within_cell_hard / cross_cell_hard / right_bowl_y1_y2_hard from hard
  rows after the fact (they are subsets of hard).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CT_DIR.parents[1]
RUN_DIR = CT_DIR / "runs" / "20260818T065918Z_t2_train_place_container_plate"
EVAL_DIR = RUN_DIR / "eval"

# Paired Q12 first-wave: answer Cover vs Zero / Self-Diverse ASAP.
PRIORITY_POINTS = ("Expert-Cover-12", "Zero", "Self-Diverse-12")
PRIORITY_SEEDS = tuple(range(8))  # 00..07
ROLLOUT_SPLITS = ("easy", "medium", "hard", "memorization_hard")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data, *, sort_keys: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=sort_keys)
        f.write("\n")


def parse_gpus(text: str) -> list[int]:
    return [int(tok) for tok in text.replace(",", " ").split() if tok.strip()]


def make_panel_splits(panels_file: Path, groups_file: Path, out_file: Path) -> Path:
    """Write ONLY unique rollout panels. Hard subsets are derived later."""
    panels = load_json(panels_file)["panels"]
    groups = load_json(groups_file)["groups"]
    payload = {
        "easy": [int(x) for x in panels["easy"]],
        "medium": [int(x) for x in panels["medium"]],
        "hard": [int(x) for x in panels["hard"]],
        # Memorization diagnostic only (excluded from primary inference).
        "memorization_hard": [int(x) for x in groups["hard"]],
    }
    # Keep insertion order for evaluation work-item order.
    write_json(out_file, payload, sort_keys=False)
    # Side file for post-hoc decomposition (not rolled out).
    derived = {
        "within_cell_hard": [int(x) for x in panels["within_cell_hard"]],
        "cross_cell_hard": [int(x) for x in panels["cross_cell_hard"]],
        "right_bowl_y1_y2_hard": [int(x) for x in panels["right_bowl_y1_y2_hard"]],
        "note": "Derived from hard rows by seed membership; do not roll out separately.",
    }
    write_json(out_file.with_name("t2_eval_derived_hard_splits.json"), derived, sort_keys=False)
    return out_file


def ckpt_for_job(job_dir: Path) -> Path:
    seed = int(job_dir.name.split("seed", 1)[1])
    return job_dir / "checkpoints" / f"t2-{seed}" / "1.ckpt"


def eval_variant(job_dir: Path) -> str:
    return f"t2_eval_{job_dir.name}"


def eval_result_path(task: str, variant: str, output_dir: Path) -> Path:
    return output_dir / task / f"{variant}.json"


def priority_job_names() -> list[str]:
    names = []
    for point in PRIORITY_POINTS:
        for seed in PRIORITY_SEEDS:
            names.append(f"{point}_seed{seed:02d}")
    return names


def job_sort_key(job_dir: Path, priority_names: list[str]) -> tuple[int, str]:
    name = job_dir.name
    if name in priority_names:
        return (0, f"{priority_names.index(name):04d}")
    return (1, name)


def run_one_eval(
    task: str,
    task_config: str,
    seeds_file: Path,
    ckpt: Path,
    variant: str,
    output_dir: Path,
    split_file: Path,
    workers_per_gpu: int,
    python_bin: str,
) -> list[str]:
    return [
        python_bin,
        str(CT_DIR / "run_eval_group.py"),
        "--workers",
        str(workers_per_gpu),
        "--",
        python_bin,
        str(CT_DIR / "eval_per_seed.py"),
        "--task",
        task,
        "--task-config",
        task_config,
        "--variant",
        variant,
        "--ckpt-path",
        str(ckpt),
        "--seeds-file",
        str(seeds_file),
        "--id-repeats",
        "0",
        "--train-repeats",
        "0",
        "--hard-repeats",
        "0",
        "--no-include-hard",
        "--extra-splits-file",
        str(split_file),
        "--extra-split-repeats",
        "8",
        "--policy-seed-offset",
        "8000",
        "--output-dir",
        str(output_dir),
    ]


def pick_python() -> str:
    conda = Path("/root/miniconda/envs/RoboTwin/bin/python")
    if conda.is_file():
        return str(conda)
    return sys.executable


def migrate_shards_for_slim_splits(task_dir: Path, split_file: Path, workers: int = 3) -> dict:
    """Rewrite existing shard JSONs for slim unique-panel rollout.

    - Keep easy/medium/hard/memorization rows.
    - Remap completed cross_cell_hard rows into hard when missing.
    - Drop within/cross/right as standalone rollout labels.
    """
    splits = load_json(split_file)
    keep = set(ROLLOUT_SPLITS)
    hard_seeds = set(int(s) for s in splits["hard"])
    stats = {"files": 0, "kept_rows": 0, "remapped_cross_to_hard": 0, "dropped_rows": 0}
    for path in sorted(task_dir.glob("*_shard_*_of_03.json")):
        payload = load_json(path)
        rows = payload.get("rows", [])
        hard_keys = {
            (int(r["env_seed"]), int(r["repeat"]))
            for r in rows
            if r.get("split") == "hard"
        }
        new_rows = []
        for row in rows:
            split = str(row.get("split"))
            if split in keep:
                new_rows.append(row)
                continue
            if split == "cross_cell_hard":
                seed = int(row["env_seed"])
                repeat = int(row["repeat"])
                if seed in hard_seeds and (seed, repeat) not in hard_keys:
                    remapped = dict(row)
                    remapped["split"] = "hard"
                    remapped["remapped_from"] = "cross_cell_hard"
                    new_rows.append(remapped)
                    hard_keys.add((seed, repeat))
                    stats["remapped_cross_to_hard"] += 1
                else:
                    stats["dropped_rows"] += 1
                continue
            stats["dropped_rows"] += 1
        new_rows.sort(key=lambda r: (r["split"], int(r["env_seed"]), int(r["repeat"])))
        stats["kept_rows"] += len(new_rows)
        stats["files"] += 1

        # Rebuild lightweight progress; completeness is re-checked by resume.
        expected = 0
        # Approximate expected for this shard: total unique episodes / workers.
        n_total = sum(len(v) * 8 for v in splits.values())
        # Same sharding as build_work_items index % workers.
        # We do not recompute exact expected here; leave complete=False unless merged.
        payload["rows"] = new_rows
        progress = dict(payload.get("progress") or {})
        progress["completed_episodes"] = len(new_rows)
        progress["complete"] = False
        progress["slim_splits_migration"] = True
        progress["expected_unique_episodes_per_job"] = n_total
        payload["progress"] = progress
        # Drop stale split summaries; resume/merge will rewrite.
        payload["splits"] = {}
        write_json(path, payload, sort_keys=False)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--gpus", default=os.environ.get("T2_EVAL_GPU_IDS", "0 1 2 3 4 5 6 7"))
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument(
        "--only-priority",
        action="store_true",
        help="Only run Cover-12/Zero/Self-Diverse seed00-07 (24 jobs).",
    )
    parser.add_argument(
        "--migrate-existing-shards",
        action="store_true",
        help="Rewrite existing shard JSONs to slim unique-panel schema before launch.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    launch = load_json(CT_DIR / "t2_launch_config.json")
    task = str(launch["task"])
    task_config = str(launch["task_config"])
    seeds_file = CT_DIR / "seeds" / f"{task}_seeds.json"
    panels_file = REPO_ROOT / launch["evaluation"]["panels_file"]
    groups_file = CT_DIR / f"difficulty_groups.{task}.v1.json"
    split_file = args.output_dir / "t2_eval_splits.json"
    make_panel_splits(panels_file, groups_file, split_file)

    gpu_ids = parse_gpus(args.gpus)
    if not gpu_ids:
        raise SystemExit("no GPU IDs")
    python_bin = pick_python()

    jobs_root = args.run_dir / "jobs"
    if not jobs_root.is_dir():
        raise SystemExit(f"missing jobs dir: {jobs_root}")

    priority_names = priority_job_names()
    job_dirs = sorted(
        (p for p in jobs_root.iterdir() if p.is_dir()),
        key=lambda p: job_sort_key(p, priority_names),
    )
    if args.only_priority:
        job_dirs = [p for p in job_dirs if p.name in priority_names]
    if args.max_jobs is not None:
        job_dirs = job_dirs[: int(args.max_jobs)]

    task_dir = args.output_dir / task
    migration_stats = None
    if args.migrate_existing_shards and task_dir.is_dir():
        migration_stats = migrate_shards_for_slim_splits(task_dir, split_file, args.workers_per_gpu)
        print(f"migrated shards: {migration_stats}", flush=True)

    pending: list[tuple[Path, Path, str]] = []
    skipped_complete = 0
    for job_dir in job_dirs:
        ckpt = ckpt_for_job(job_dir)
        if not ckpt.is_file():
            raise SystemExit(f"missing checkpoint: {ckpt}")
        variant = eval_variant(job_dir)
        out = eval_result_path(task, variant, args.output_dir)
        if out.is_file():
            # Still enqueue; run_eval_group will reuse if complete.
            pending.append((job_dir, ckpt, variant))
            skipped_complete += 0
        else:
            pending.append((job_dir, ckpt, variant))

    n_unique = sum(len(load_json(split_file)[k]) for k in ROLLOUT_SPLITS)
    meta = {
        "record": "capability_transport.t2_eval.launch.v2_slim_priority",
        "created_at": utc_now(),
        "run_dir": str(args.run_dir),
        "output_dir": str(args.output_dir),
        "task": task,
        "task_config": task_config,
        "jobs": len(job_dirs),
        "pending": len(pending),
        "gpus": gpu_ids,
        "workers_per_gpu": int(args.workers_per_gpu),
        "split_file": str(split_file),
        "rollout_splits": list(ROLLOUT_SPLITS),
        "unique_seeds_per_job": n_unique,
        "episodes_per_job": n_unique * 8,
        "policy_seed_offset": 8000,
        "repeats_per_seed": 8,
        "python_bin": python_bin,
        "only_priority": bool(args.only_priority),
        "priority_jobs": priority_names if args.only_priority else priority_names,
        "migration_stats": migration_stats,
        "note": (
            "within/cross/right_bowl hard metrics are derived from hard rows; "
            "not rolled out as separate splits."
        ),
    }
    write_json(args.output_dir / "launch_meta.json", meta)

    print(f"run_dir: {args.run_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"jobs: {len(job_dirs)} pending: {len(pending)} only_priority={args.only_priority}")
    print(f"unique_seeds/job={n_unique} episodes/job={n_unique * 8}")
    print(f"gpus: {gpu_ids} workers_per_gpu={args.workers_per_gpu}")
    if args.dry_run:
        for job_dir, ckpt, variant in pending:
            print(f"would eval {job_dir.name} -> {variant}")
        return 0

    slots: dict[int, tuple[subprocess.Popen, str] | None] = {gpu: None for gpu in gpu_ids}
    queue = list(pending)
    failures: list[str] = []
    started = 0
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env_base["PYTHONPATH"] if env_base.get("PYTHONPATH") else ""
    )

    while queue or any(v is not None for v in slots.values()):
        for gpu, slot in list(slots.items()):
            if slot is None:
                continue
            proc, name = slot
            rc = proc.poll()
            if rc is None:
                continue
            if rc == 0:
                print(f"GPU {gpu}: completed {name}", flush=True)
            else:
                print(f"GPU {gpu}: FAILED {name} exit={rc}", flush=True)
                failures.append(name)
            slots[gpu] = None
        for gpu, slot in list(slots.items()):
            if slot is not None or not queue:
                continue
            job_dir, ckpt, variant = queue.pop(0)
            cmd = run_one_eval(
                task=task,
                task_config=task_config,
                seeds_file=seeds_file,
                ckpt=ckpt,
                variant=variant,
                output_dir=args.output_dir,
                split_file=split_file,
                workers_per_gpu=args.workers_per_gpu,
                python_bin=python_bin,
            )
            env = env_base.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_path = args.output_dir / "logs" / f"{variant}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"# start {utc_now()} gpu={gpu} slim_v2\n")
                f.write(" ".join(cmd) + "\n")
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env)
            slots[gpu] = (proc, job_dir.name)
            started += 1
            print(f"GPU {gpu}: start {job_dir.name}", flush=True)
        time.sleep(5)

    meta["finished_at"] = utc_now()
    meta["started"] = started
    meta["failures"] = failures
    meta["status"] = "failed" if failures else "complete"
    write_json(args.output_dir / "launch_meta.json", meta)
    if failures:
        print(f"eval finished with failures={len(failures)}", flush=True)
        return 1
    print("all eval jobs completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
