#!/usr/bin/env python3
"""Schedule the frozen 72 T2 training jobs: 1 DP trainer per GPU, skip complete.

Does not start jobs unless invoked. Resume by re-running the same --run-dir.

Usage (inside the cloud container, from /workspace/RoboTwin):
    python experiments/capability_transport/launch_t2_train.py --dry-run
    python experiments/capability_transport/launch_t2_train.py --build-only
    python experiments/capability_transport/launch_t2_train.py --run-dir <dir>
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CT_DIR))

from common import append_jsonl, repo_path  # noqa: E402
from t2_train_lib import (  # noqa: E402
    DEFAULT_MIXTURE_DIR,
    checkpoint_path,
    iter_jobs,
    job_id,
    load_launch_config,
    mixture_zarr_path,
    python_bin,
    write_json,
)

LEDGER = CT_DIR / "efficiency_ledger.jsonl"
LATEST_PREFIX = "LATEST_T2_TRAIN_"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_gpu_ids(text: str) -> List[int]:
    return [int(tok) for tok in text.replace(",", " ").split() if tok.strip()]


def occupied_gpu_ids() -> Dict[int, List[str]]:
    """Map GPU index -> compute-process command lines, empty if nvidia-smi fails."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        uuid_to_idx = {}
        idx_proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        for line in idx_proc.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                uuid_to_idx[parts[1]] = int(parts[0])
        occupied: Dict[int, List[str]] = {}
        for line in proc.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            gpu_uuid, pid, name = parts[0], parts[1], parts[2]
            idx = uuid_to_idx.get(gpu_uuid)
            if idx is None:
                continue
            occupied.setdefault(idx, []).append(f"{name} pid={pid}")
        return occupied
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}


def default_run_dir(task: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return CT_DIR / "runs" / f"{stamp}_t2_train_{task}"


def resolve_run_dir(task: str, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return Path(explicit)
    latest = CT_DIR / "runs" / f"{LATEST_PREFIX}{task}"
    if latest.is_file():
        stored = Path(latest.read_text().strip())
        if stored.is_dir():
            return stored
    return default_run_dir(task)


def job_complete(run_dir: Path, point: str, seed: int) -> bool:
    return checkpoint_path(run_dir / "jobs" / job_id(point, seed), seed).is_file()


def build_mixtures(mixture_dir: Path, points: Optional[List[str]] = None) -> None:
    cmd = [python_bin(), str(CT_DIR / "build_t2_mixtures.py"), "--mixture-dir", str(mixture_dir)]
    if points:
        for point in points:
            cmd.extend(["--point", point])
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(repo_path()))


def launch_job(point: str, seed: int, run_dir: Path, mixture_dir: Path, gpu: int) -> subprocess.Popen:
    job_dir = run_dir / "jobs" / job_id(point, seed)
    job_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_bin(),
        str(CT_DIR / "train_t2_job.py"),
        "--point",
        point,
        "--seed",
        str(int(seed)),
        "--run-dir",
        str(run_dir),
        "--mixture-dir",
        str(mixture_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["HYDRA_FULL_ERROR"] = "1"
    repo = str(repo_path())
    env["PYTHONPATH"] = repo + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    print(f"GPU {gpu}: start {job_id(point, seed)}", flush=True)
    return subprocess.Popen(cmd, cwd=str(repo_path()), env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch frozen T2 72-job training grid")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--mixture-dir", type=Path, default=DEFAULT_MIXTURE_DIR)
    parser.add_argument("--gpus", default=os.environ.get("T2_TRAIN_GPU_IDS", "0 1 2 3 4 5 6 7"))
    parser.add_argument("--point", action="append", dest="points", default=None)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--force-gpus",
        action="store_true",
        help="do not skip GPUs that already have compute processes",
    )
    args = parser.parse_args()

    cfg, launch_sha = load_launch_config()
    task = cfg["task"]
    n_jobs = int(cfg["training_grid"]["n_jobs"])
    gpu_ids = parse_gpu_ids(args.gpus)
    if not gpu_ids:
        raise SystemExit("no GPU ids")

    occupied = occupied_gpu_ids()
    free_gpus = []
    for gpu in gpu_ids:
        procs = occupied.get(gpu, [])
        if procs and not args.force_gpus:
            print(f"GPU {gpu} busy ({'; '.join(procs)}); skipping this GPU")
            continue
        free_gpus.append(gpu)
    if not args.dry_run and not args.build_only and not free_gpus:
        raise SystemExit("no free GPUs; pass --force-gpus only if you intend to colocate")

    run_dir = resolve_run_dir(task, args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run and not args.build_only:
        latest = CT_DIR / "runs" / f"{LATEST_PREFIX}{task}"
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_text(str(run_dir) + "\n")

    jobs = iter_jobs(cfg)
    if args.points:
        jobs = [(p, s) for p, s in jobs if p in args.points]
    pending: List[Tuple[str, int]] = []
    completed = 0
    for point, seed in jobs:
        if job_complete(run_dir, point, seed):
            completed += 1
            continue
        pending.append((point, seed))
    if args.max_jobs is not None:
        pending = pending[: int(args.max_jobs)]

    run_meta = {
        "record": "capability_transport.t2_train_run.v1",
        "created_at": utc_now(),
        "task": task,
        "run_dir": str(run_dir),
        "launch_config_sha256": launch_sha,
        "protocol_sha256": cfg["depends_on"]["protocol.t2_training.v1.json"],
        "n_jobs_grid": n_jobs,
        "n_jobs_selected": len(jobs),
        "n_already_complete": completed,
        "n_pending": len(pending),
        "gpu_ids_requested": gpu_ids,
        "gpu_ids_free": free_gpus,
        "mixture_dir": str(args.mixture_dir),
        "dry_run": bool(args.dry_run),
        "build_only": bool(args.build_only),
    }
    write_json(run_dir / "run_meta.json", run_meta)
    print(f"run_dir {run_dir}")
    print(f"launch_config {launch_sha}")
    print(f"grid {n_jobs}; selected {len(jobs)}; complete {completed}; pending {len(pending)}")
    print(f"GPUs requested {gpu_ids}; free {free_gpus}")

    if args.dry_run:
        for point, seed in pending:
            print(f"  would train {job_id(point, seed)}")
        return

    args.mixture_dir.mkdir(parents=True, exist_ok=True)
    build_points = sorted({p for p, _ in jobs})
    build_mixtures(args.mixture_dir, build_points)
    if args.build_only:
        print("mixtures ready; not launching trainers")
        return

    append_jsonl(
        LEDGER,
        {
            "stage": "t2_train",
            "task": task,
            "created_at": utc_now(),
            "record": "capability_transport.t2_train.v1",
            "run_dir": str(run_dir),
            "launch_config_sha256": launch_sha,
            "n_jobs": len(pending),
            "n_already_complete": completed,
            "gpus": free_gpus,
            "status": "launched",
            "training_runs": len(pending),
            "gradient_steps_budget_per_job": cfg["matched_training"]["s_star"],
        },
    )

    slots: Dict[int, Optional[Tuple[subprocess.Popen, str, int]]] = {gpu: None for gpu in free_gpus}
    queue = list(pending)
    failures: List[str] = []
    started = 0
    while queue or any(slot is not None for slot in slots.values()):
        for gpu, slot in list(slots.items()):
            if slot is None:
                continue
            proc, point, seed = slot
            rc = proc.poll()
            if rc is None:
                continue
            name = job_id(point, seed)
            if rc == 0 and job_complete(run_dir, point, seed):
                print(f"GPU {gpu}: completed {name}", flush=True)
            else:
                print(f"GPU {gpu}: FAILED {name} exit={rc}", flush=True)
                failures.append(name)
            slots[gpu] = None
        for gpu, slot in list(slots.items()):
            if slot is not None or not queue:
                continue
            point, seed = queue.pop(0)
            slots[gpu] = (launch_job(point, seed, run_dir, args.mixture_dir, gpu), point, seed)
            started += 1
        time.sleep(5)

    run_meta["finished_at"] = utc_now()
    run_meta["n_started"] = started
    run_meta["failures"] = failures
    run_meta["status"] = "failed" if failures else "complete"
    write_json(run_dir / "run_meta.json", run_meta)
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed: {failures}")
    print(f"T2 training complete: {run_dir}")


if __name__ == "__main__":
    main()
