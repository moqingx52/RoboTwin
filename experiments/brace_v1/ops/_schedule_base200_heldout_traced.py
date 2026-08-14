#!/usr/bin/env python3
"""Collect Base200 traced rollouts for held-out tasks that already have 600.ckpt.

Skips place_container_plate (already complete) and stack_bowls_three (parked).
Does not freeze BRACE-v2 or launch confirmatory U0/N1/B1/B2/B3 arms.
Packing: 3 simulator workers per GPU on GPUs 0-7.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path("/workspace/RoboTwin")
PY = "/root/miniconda/envs/RoboTwin/bin/python"
PROTOCOL = REPO / "experiments/brace/multitask_protocol.v2.json"
SEED_DIR = REPO / "experiments/brace/seeds/multitask_v1"
ROLLOUT_DIR = REPO / "experiments/brace/rollouts_traced_base200_v2"
LOG_DIR = REPO / "experiments/brace/logs/traced_base200_heldout"
STATE_PATH = REPO / "experiments/brace/runs/base200_heldout_traced_state.json"
GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
WORKERS_PER_GPU = 3
NUM_SHARDS = 15
ROLLOUTS_PER_SEED = 8
TASKS = [
    "beat_block_hammer",
    "click_alarmclock",
    "handover_mic",
    "lift_pot",
    "move_can_pot",
    "open_laptop",
    "place_burger_fries",
    "put_object_cabinet",
    "shake_bottle",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_seeds(task: str) -> Path:
    manifest_path = SEED_DIR / f"{task}.json"
    manifest = read_json(manifest_path)
    seeds = [int(x) for x in manifest["partitions"]["rollout_train"]]
    if len(seeds) != 100:
        raise SystemExit(f"{task}: expected 100 rollout_train seeds, got {len(seeds)}")
    train = {int(x) for x in (REPO / f"data/{task}/demo_clean/seed.txt").read_text().split()}
    parts = manifest["partitions"]
    seed_set = set(seeds)
    for name in ("confirm_easy", "confirm_hard", "census_candidate", "anchor_candidate"):
        other = {int(x) for x in parts[name]}
        overlap = seed_set & other
        if overlap:
            raise SystemExit(f"{task}: rollout_train overlaps {name}: {sorted(overlap)[:5]}")
    overlap_train = seed_set & train
    if overlap_train:
        raise SystemExit(f"{task}: rollout_train overlaps Base200 train seeds: {sorted(overlap_train)[:5]}")
    out = ROLLOUT_DIR / task / "base200_v2_rollout_train_seeds.json"
    write_json(
        out,
        {
            "schema_version": 1,
            "task": task,
            "split": "rollout_train",
            "train_rollout": seeds,
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": file_sha256(manifest_path),
            "protocol": str(PROTOCOL),
            "protocol_sha256": file_sha256(PROTOCOL),
            "notes": "Held-out Base200 traced corpus from multitask_v1 partitions.rollout_train. stack_bowls_three parked.",
        },
    )
    return out


def shard_done(task: str, shard_id: int) -> bool:
    task_dir = ROLLOUT_DIR / task
    stats = task_dir / f"seed_stats_shard_{shard_id:02d}_of_{NUM_SHARDS:02d}.json"
    manifest = task_dir / f"manifest_shard_{shard_id:02d}_of_{NUM_SHARDS:02d}.jsonl"
    return stats.is_file() and manifest.is_file()


def launch_shard(task: str, gpu: int, shard_id: int, seeds_file: Path) -> subprocess.Popen:
    log_path = LOG_DIR / f"{task}_shard{shard_id:02d}_of_{NUM_SHARDS:02d}.log"
    cmd = [
        PY,
        "experiments/brace/collect_traced_rollouts.py",
        "--task",
        task,
        "--task-config",
        "demo_brace_trace",
        "--seeds-file",
        str(seeds_file),
        "--output-dir",
        str(ROLLOUT_DIR),
        "--expert-data-num",
        "200",
        "--checkpoint-num",
        "600",
        "--train-seed",
        "0",
        "--rollouts-per-seed",
        str(ROLLOUTS_PER_SEED),
        "--action-dim",
        "14",
        "--shard-id",
        str(shard_id),
        "--num-shards",
        str(NUM_SHARDS),
        "--snapshots-per-trajectory",
        "3",
        "--save-failures",
        "--resume",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = f"{REPO}:{REPO}/policy/DP"
    proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    proc._brace_log_f = handle  # type: ignore[attr-defined]
    return proc


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        return False
    try:
        state = (proc / "stat").read_text().split(")")[-1].split()[0]
        return state != "Z"
    except OSError:
        return False


def save(state: dict[str, Any]) -> None:
    write_json(STATE_PATH, state)


def main() -> int:
    os.chdir(REPO)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ROLLOUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_files = {task: materialize_seeds(task) for task in TASKS}
    if STATE_PATH.is_file():
        state = read_json(STATE_PATH)
    else:
        state = {
            "schema_version": 1,
            "kind": "base200_heldout_traced",
            "skipped": {
                "place_container_plate": "already_complete",
                "stack_bowls_three": "parked_incomplete_base200_train",
            },
            "num_shards": NUM_SHARDS,
            "workers_per_gpu": WORKERS_PER_GPU,
            "gpus": GPUS,
            "jobs": {
                f"{task}:{shard}": {"task": task, "shard": shard, "status": "pending"}
                for task in TASKS
                for shard in range(NUM_SHARDS)
            },
        }
    for key, job in state["jobs"].items():
        task, shard = job["task"], int(job["shard"])
        if shard_done(task, shard):
            job["status"] = "completed"
            job.pop("pid", None)
            job.pop("gpu", None)
        elif job.get("status") == "running" and not pid_alive(job.get("pid")):
            job["status"] = "pending"
            job.pop("pid", None)
            job.pop("gpu", None)
    save(state)
    print(f"held-out traced scheduler tasks={TASKS} shards={NUM_SHARDS} gpus={GPUS}", flush=True)

    running: dict[str, dict[str, Any]] = {}
    while True:
        for key, slot in list(running.items()):
            proc = slot["proc"]
            job = slot["job"]
            if proc.poll() is None:
                continue
            handle = getattr(proc, "_brace_log_f", None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
            if proc.returncode == 0 and shard_done(job["task"], int(job["shard"])):
                job["status"] = "completed"
                print(f"[completed] {key}", flush=True)
            else:
                job["status"] = "pending"
                job["error"] = f"exit={proc.returncode}"
                print(f"[retry] {key} exit={proc.returncode}", flush=True)
            job.pop("pid", None)
            job.pop("gpu", None)
            running.pop(key, None)

        gpu_load = {gpu: 0 for gpu in GPUS}
        for slot in running.values():
            gpu_load[slot["gpu"]] += 1
        pending = [j for j in state["jobs"].values() if j["status"] == "pending"]
        for job in pending:
            free = [g for g in GPUS if gpu_load[g] < WORKERS_PER_GPU]
            if not free:
                break
            gpu = free[0]
            key = f"{job['task']}:{job['shard']}"
            proc = launch_shard(job["task"], gpu, int(job["shard"]), seed_files[job["task"]])
            job["status"] = "running"
            job["gpu"] = gpu
            job["pid"] = proc.pid
            gpu_load[gpu] += 1
            running[key] = {"proc": proc, "job": job, "gpu": gpu}
            print(f"[running] {key} gpu={gpu} pid={proc.pid}", flush=True)

        n_done = sum(1 for j in state["jobs"].values() if j["status"] == "completed")
        n_run = sum(1 for j in state["jobs"].values() if j["status"] == "running")
        n_pend = sum(1 for j in state["jobs"].values() if j["status"] == "pending")
        save(state)
        print(f"progress completed={n_done} running={n_run} pending={n_pend}", flush=True)
        if n_pend == 0 and n_run == 0:
            print("all held-out traced shards launched/completed", flush=True)
            return 0
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
