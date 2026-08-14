#!/usr/bin/env python3
"""One-off: run B3 seed5 confirm_hard across 8 GPUs × 3 shards, then merge.

Does not touch the adaptation scheduler. Intended after the scheduler is stopped
and partial *_of_03 hard shards have been deleted.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/workspace/RoboTwin")
PY = "/root/miniconda/envs/RoboTwin/bin/python"
PHASE1 = REPO / "experiments" / "phase1"
BRACE = REPO / "experiments" / "brace"
RUN = REPO / "experiments/brace/runs/20260812T031200Z_place_base200_v2_line_a_pilot"
VARIANT = "line_a_B3_seed5_confirm_hard"
TASK = "place_container_plate"
GPUS = list(range(8))
WORKERS_PER_GPU = 3
NUM_SHARDS = len(GPUS) * WORKERS_PER_GPU


def main() -> int:
    os.chdir(REPO)
    out_dir = RUN / "eval"
    task_dir = out_dir / TASK
    log_dir = RUN / "logs"
    ckpt = REPO / "policy/DP/checkpoints/place_container_plate-brace-place_base200_v2-B3-5/10.ckpt"
    seeds = RUN / "eval_seeds/confirm_hard.json"
    leaf = [
        PY,
        str(PHASE1 / "eval_per_seed.py"),
        "--task",
        TASK,
        "--task-config",
        "demo_randomized",
        "--variant",
        VARIANT,
        "--ckpt-path",
        str(ckpt),
        "--output-dir",
        str(out_dir),
        "--seeds-file",
        str(seeds),
        "--id-repeats",
        "1",
        "--train-seed-count",
        "0",
        "--no-include-hard",
        "--resume",
        "--num-shards",
        str(NUM_SHARDS),
    ]
    procs: list[tuple[int, int, subprocess.Popen]] = []
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = f"{REPO}:{REPO}/policy/DP"
    for gpu in GPUS:
        for local in range(WORKERS_PER_GPU):
            shard = gpu * WORKERS_PER_GPU + local
            env = env_base.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_path = log_dir / f"eval_B3_seed5_confirm_hard_shard_{shard:02d}_of_{NUM_SHARDS:02d}.log"
            handle = log_path.open("w", encoding="utf-8")
            cmd = leaf + ["--shard-id", str(shard)]
            proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=handle, stderr=subprocess.STDOUT)
            procs.append((gpu, shard, proc))
            print(f"launch shard={shard:02d} gpu={gpu} pid={proc.pid}", flush=True)
    failed = []
    while True:
        alive = [(gpu, shard, proc) for gpu, shard, proc in procs if proc.poll() is None]
        done = [(gpu, shard, proc) for gpu, shard, proc in procs if proc.poll() is not None]
        bad = [(gpu, shard, proc.returncode) for gpu, shard, proc in done if proc.returncode != 0]
        print(
            f"[{time.strftime('%H:%M:%S')}] alive={len(alive)} done={len(done)-len(bad)} failed={len(bad)}",
            flush=True,
        )
        if bad and not failed:
            failed = bad
        if not alive:
            break
        time.sleep(30)
    if failed:
        print(f"shard failures: {failed}", flush=True)
        return 1
    merge = [
        PY,
        str(PHASE1 / "merge_eval_shards.py"),
        "--task",
        TASK,
        "--task-config",
        "demo_randomized",
        "--variant",
        VARIANT,
        "--output-dir",
        str(out_dir),
        "--num-shards",
        str(NUM_SHARDS),
    ]
    proc = subprocess.run(merge, cwd=str(REPO), check=False)
    if proc.returncode != 0:
        print(f"merge failed exit={proc.returncode}", flush=True)
        return proc.returncode
    merged = task_dir / f"{VARIANT}.json"
    payload = json.loads(merged.read_text(encoding="utf-8"))
    complete = bool((payload.get("progress") or {}).get("complete"))
    n_rows = len(payload.get("rows") or [])
    print(f"merged {merged} complete={complete} n_rows={n_rows}", flush=True)
    if not complete or n_rows != 100:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
