#!/usr/bin/env python3
"""Schedule place Base200 Line A pilot: U0/N1/B1/B2/B3 x training seeds 1-5.

Uses exclusive 1 DP trainer per GPU on the free pool (default 0 1 2 4 5 6 7).
Never schedules onto GPU 3 while stack_bowls_three training is reserved.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

REPO = Path("/workspace/RoboTwin")
PY = "/root/miniconda/envs/RoboTwin/bin/python"
METHODS = ("U0", "N1", "B1", "B2", "B3")
SEEDS = (1, 2, 3, 4, 5)
DEFAULT_GPUS = (0, 1, 2, 4, 5, 6, 7)
RESERVED_STACK_GPU = 3
# Set BRACE_ALLOW_GPU3=1 to include GPU 3 in the pool (e.g. stack idle at 300.ckpt).
EPOCHS = 10
BATCH_SIZE = 128
LR = "0.00005"
ROLLOUT_PER_BATCH = 16
CHECKPOINT_LABEL = "place_base200_v2"
SCREEN_PROTOCOL = "experiments/brace/screen_protocol.v1.2.json"


def compute_steps(expert: Path, batch_size: int) -> int:
    import zarr

    root = zarr.open(str(expert), mode="r")
    frames = int(root["meta/episode_ends"][-1])
    steps = frames // batch_size
    if steps < 1:
        raise RuntimeError(f"expert too small: {expert}")
    return steps


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        return False
    try:
        # Reap zombies we parented; treat Z as not alive.
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    if not proc.exists():
        return False
    try:
        stat = (proc / "stat").read_text().split()
        # Field 3 is state after comm; but comm can contain spaces in ().
        # Parse from the last ')' then next token is state.
        raw = (proc / "stat").read_text()
        state = raw.split(")")[-1].split()[0]
        if state == "Z":
            return False
    except OSError:
        return False
    return True


def main() -> int:
    os.chdir(REPO)
    run_ptr = REPO / "experiments/brace/runs/LATEST_place_base200_v2_line_a_pilot"
    run_dir = REPO / run_ptr.read_text(encoding="utf-8").strip()
    datasets = run_dir / "datasets"
    logs = run_dir / "logs"
    state_path = run_dir / "train_state.json"
    logs.mkdir(parents=True, exist_ok=True)

    expert = REPO / "policy/DP/data/place_container_plate-demo_clean-200.zarr"
    steps = compute_steps(expert, BATCH_SIZE)
    dataset_for = {
        "U0": datasets / "place_container_plate_U0.zarr",
        "N1": datasets / "place_container_plate_N1.zarr",
        "B1": datasets / "place_container_plate_B1.zarr",
        "B2": datasets / "place_container_plate_N1.zarr",
        "B3": datasets / "place_container_plate_B1.zarr",
    }
    anchor = datasets / "place_container_plate_anchor_replay.zarr"
    for method, path in dataset_for.items():
        if not path.exists():
            raise SystemExit(f"missing dataset for {method}: {path}")
    if not anchor.is_dir():
        raise SystemExit(f"missing anchor: {anchor}")

    gpu_ids = [int(x) for x in os.environ.get("BRACE_GPU_IDS", " ".join(map(str, DEFAULT_GPUS))).split()]
    allow_gpu3 = os.environ.get("BRACE_ALLOW_GPU3", "0") == "1"
    if allow_gpu3 and RESERVED_STACK_GPU not in gpu_ids:
        gpu_ids = sorted(set(gpu_ids) | {RESERVED_STACK_GPU})
    if RESERVED_STACK_GPU in gpu_ids and not allow_gpu3:
        raise SystemExit(f"refusing to schedule on reserved stack GPU {RESERVED_STACK_GPU}")

    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        jobs = []
        for method in METHODS:
            for seed in SEEDS:
                jobs.append(
                    {
                        "id": f"train:{method}:seed{seed}",
                        "method": method,
                        "seed": seed,
                        "status": "pending",
                        "gpu": None,
                        "pid": None,
                        "log": str(logs / f"train_{method}_seed{seed}.log"),
                        "attempts": 0,
                    }
                )
        state = {
            "schema_version": 1,
            "run_dir": str(run_dir),
            "checkpoint_label": CHECKPOINT_LABEL,
            "epochs": EPOCHS,
            "steps_per_epoch": steps,
            "gpus": gpu_ids,
            "reserved_gpu": RESERVED_STACK_GPU,
            "jobs": jobs,
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def save() -> None:
        if state_path.is_file():
            on_disk = json.loads(state_path.read_text(encoding="utf-8"))
            preserve_keys = (
                "note",
                "artifact_reconciled_at_utc",
                "artifact_reconcile_report",
                "checkpoint_sha256",
                "reconcile_prior",
            )
            disk_by_id = {job["id"]: job for job in on_disk.get("jobs", [])}
            for job in state["jobs"]:
                if job.get("status") != "completed":
                    continue
                disk_job = disk_by_id.get(job["id"], {})
                if disk_job.get("note") == "artifact_reconciled_after_post_training_failure":
                    for key in preserve_keys:
                        if key in disk_job:
                            job[key] = disk_job[key]
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    save()
    print(
        f"scheduling {len(state['jobs'])} jobs on gpus={gpu_ids} "
        f"steps_per_epoch={steps} epochs={EPOCHS} run={run_dir}",
        flush=True,
    )

    running: dict[int, dict] = {}
    # Resume: adopt still-alive running jobs from prior scheduler instance.
    for job in state["jobs"]:
        if job.get("status") != "running":
            continue
        gpu = job.get("gpu")
        pid = job.get("pid")
        if gpu is None:
            continue
        if pid_alive(pid):
            running[int(gpu)] = job
            print(f"[adopt] {job['id']} gpu={gpu} pid={pid}", flush=True)
        else:
            ckpt = (
                REPO
                / "policy/DP/checkpoints"
                / f"place_container_plate-brace-{CHECKPOINT_LABEL}-{job['method']}-{job['seed']}"
                / f"{EPOCHS}.ckpt.complete"
            )
            job["status"] = "completed" if ckpt.is_file() else "failed"
            job["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            print(f"[{job['status']}] {job['id']} gpu={gpu} on-resume", flush=True)

    def poll_running() -> None:
        done = []
        for gpu, job in list(running.items()):
            pid = job.get("pid")
            if pid_alive(pid):
                continue
            ckpt = (
                REPO
                / "policy/DP/checkpoints"
                / f"place_container_plate-brace-{CHECKPOINT_LABEL}-{job['method']}-{job['seed']}"
                / f"{EPOCHS}.ckpt.complete"
            )
            job["status"] = "completed" if ckpt.is_file() else "failed"
            job["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            print(f"[{job['status']}] {job['id']} gpu={gpu}", flush=True)
            done.append(gpu)
        for gpu in done:
            running.pop(gpu, None)

    def launch(job: dict, gpu: int) -> None:
        method = job["method"]
        seed = job["seed"]
        dataset = dataset_for[method]
        log_path = Path(job["log"])
        cmd = [
            "bash",
            "experiments/brace/finetune.sh",
            "place_container_plate",
            method,
            str(gpu),
            str(seed),
            str(EPOCHS),
            str(dataset),
            CHECKPOINT_LABEL,
            str(steps),
            LR,
            str(ROLLOUT_PER_BATCH),
            str(BATCH_SIZE),
            str(anchor),
            SCREEN_PROTOCOL,
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{REPO}:{REPO}/policy/DP"
        env["PATH"] = f"/root/miniconda/envs/RoboTwin/bin:{env.get('PATH', '')}"
        handle = log_path.open("a", encoding="utf-8")
        handle.write(f"\n=== launch {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} gpu={gpu} ===\n")
        handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        job["status"] = "running"
        job["gpu"] = gpu
        job["pid"] = proc.pid
        job["attempts"] = int(job.get("attempts", 0)) + 1
        job["started_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        running[gpu] = job
        print(f"[running] {job['id']} gpu={gpu} pid={proc.pid}", flush=True)

    try:
        while True:
            poll_running()
            pending = [j for j in state["jobs"] if j["status"] == "pending"]
            completed = [j for j in state["jobs"] if j["status"] == "completed"]
            failed = [j for j in state["jobs"] if j["status"] == "failed"]
            save()
            if not pending and not running:
                print(
                    f"all done completed={len(completed)} failed={len(failed)} / {len(state['jobs'])}",
                    flush=True,
                )
                return 0 if not failed else 1
            free = [g for g in gpu_ids if g not in running]
            for gpu, job in zip(free, pending):
                launch(job, gpu)
            save()
            time.sleep(30)
    except KeyboardInterrupt:
        print("interrupted; leaving running workers", flush=True)
        save()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
