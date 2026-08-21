#!/usr/bin/env python3
"""Launch T2 post-train eval for all 72 checkpoints with full GPU utilization.

Scheduling model:
- One logical eval job per GPU.
- Each logical eval job runs `run_eval_group.py --workers 3`, so each GPU hosts
  3 concurrent eval shards (as required by the measured screen policy).
- 8 GPUs -> 8 concurrent logical eval jobs.
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


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def parse_gpus(text: str) -> list[int]:
    return [int(tok) for tok in text.replace(",", " ").split() if tok.strip()]


def make_panel_splits(panels_file: Path, groups_file: Path, out_file: Path) -> Path:
    panels = load_json(panels_file)["panels"]
    groups = load_json(groups_file)["groups"]
    payload = {
        "easy": [int(x) for x in panels["easy"]],
        "medium": [int(x) for x in panels["medium"]],
        "hard": [int(x) for x in panels["hard"]],
        "within_cell_hard": [int(x) for x in panels["within_cell_hard"]],
        "cross_cell_hard": [int(x) for x in panels["cross_cell_hard"]],
        "right_bowl_y1_y2_hard": [int(x) for x in panels["right_bowl_y1_y2_hard"]],
        # Memorization diagnostic only (excluded from primary inference).
        "memorization_hard": [int(x) for x in groups["hard"]],
    }
    write_json(out_file, payload)
    return out_file


def ckpt_for_job(job_dir: Path) -> Path:
    # Job name: <Point>_seedNN ; training.seed used for checkpoint directory.
    seed = int(job_dir.name.split("seed", 1)[1])
    return job_dir / "checkpoints" / f"t2-{seed}" / "1.ckpt"


def eval_variant(job_dir: Path) -> str:
    return f"t2_eval_{job_dir.name}"


def eval_result_path(task: str, variant: str, output_dir: Path) -> Path:
    return output_dir / task / f"{variant}.json"


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--gpus", default=os.environ.get("T2_EVAL_GPU_IDS", "0 1 2 3 4 5 6 7"))
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument("--max-jobs", type=int, default=None)
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
    job_dirs = sorted(p for p in jobs_root.iterdir() if p.is_dir())
    if args.max_jobs is not None:
        job_dirs = job_dirs[: int(args.max_jobs)]

    pending: list[tuple[Path, Path, str]] = []
    for job_dir in job_dirs:
        ckpt = ckpt_for_job(job_dir)
        if not ckpt.is_file():
            raise SystemExit(f"missing checkpoint: {ckpt}")
        variant = eval_variant(job_dir)
        out = eval_result_path(task, variant, args.output_dir)
        if out.is_file():
            # run_eval_group still verifies completeness if re-run.
            pending.append((job_dir, ckpt, variant))
        else:
            pending.append((job_dir, ckpt, variant))

    meta = {
        "record": "capability_transport.t2_eval.launch.v1",
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
        "policy_seed_offset": 8000,
        "repeats_per_seed": 8,
        "python_bin": python_bin,
    }
    write_json(args.output_dir / "launch_meta.json", meta)

    print(f"run_dir: {args.run_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"jobs: {len(job_dirs)} pending: {len(pending)}")
    print(f"gpus: {gpu_ids} workers_per_gpu={args.workers_per_gpu}")
    if args.dry_run:
        for job_dir, ckpt, variant in pending[:12]:
            print(f"would eval {job_dir.name} -> {variant} ({ckpt})")
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
                f.write(f"# start {utc_now()} gpu={gpu}\n")
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

