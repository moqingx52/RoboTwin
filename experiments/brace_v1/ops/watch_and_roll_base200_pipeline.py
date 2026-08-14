#!/usr/bin/env python3
"""Roll Base200 frozen-eval backlog on free GPUs.

Traced collector owns pool {2,4,5,6,7} while shards remain pending. Former
train GPUs {0,1,3} may take backlog as soon as they are nvidia-idle (do not
interrupt an active train.py). When traced is not pending, backlog may also
use freed traced-pool GPUs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
RUN = REPO / "experiments/brace/runs/base200_frozen_eval_20260811T012027Z"
LOG = REPO / "experiments/brace/logs/base200_frozen_eval"
STATE = REPO / "experiments/brace/runs/base200_place_traced_15shard_20260811.state.json"
TRACED_POOL = [2, 4, 5, 6, 7]
BACKLOG_EXTRA = [0, 1, 3]
POOL = TRACED_POOL + BACKLOG_EXTRA

BACKLOG = [
    ("move_can_pot", "demo_clean", "base200_confirm_easy"),
    ("click_alarmclock", "demo_randomized", "base200_confirm_hard"),
    ("move_can_pot", "demo_randomized", "base200_confirm_hard"),
    ("handover_mic", "demo_randomized", "base200_confirm_hard"),
    ("open_laptop", "demo_clean", "base200_confirm_easy"),
    ("open_laptop", "demo_randomized", "base200_confirm_hard"),
    ("put_object_cabinet", "demo_clean", "base200_confirm_easy"),
    ("put_object_cabinet", "demo_randomized", "base200_confirm_hard"),
    ("place_burger_fries", "demo_clean", "base200_confirm_easy"),
    ("place_burger_fries", "demo_randomized", "base200_confirm_hard"),
    ("shake_bottle", "demo_clean", "base200_confirm_easy"),
    ("shake_bottle", "demo_randomized", "base200_confirm_hard"),
    ("stack_bowls_three", "demo_clean", "base200_confirm_easy"),
    ("stack_bowls_three", "demo_randomized", "base200_confirm_hard"),
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def ckpt_ready(task: str) -> bool:
    root = REPO / f"policy/DP/checkpoints/{task}-demo_clean-200-0"
    return (root / "600.ckpt").is_file() and (root / "brace_base200_provenance.json").is_file()


def job_complete(task: str, variant: str) -> bool:
    path = RUN / "results" / task / f"{variant}.json"
    if path.is_file() and bool((read_json(path).get("progress") or {}).get("complete")):
        return True
    # If all shards finished but run_eval_group died before merge, merge now.
    shard_paths = sorted((RUN / "results" / task).glob(f"{variant}_shard_*_of_*.json"))
    if len(shard_paths) < 3:
        return False
    num_shards = None
    for sp in shard_paths:
        prog = (read_json(sp).get("progress") or {})
        if not prog.get("complete"):
            return False
        num_shards = int(prog.get("num_shards") or 0) or num_shards
    if num_shards and len(shard_paths) == num_shards:
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "experiments/phase1/merge_eval_shards.py",
                    "--task",
                    task,
                    "--variant",
                    variant,
                    "--output-dir",
                    str(RUN / "results"),
                    "--num-shards",
                    str(num_shards),
                ],
                cwd=str(REPO),
            )
            print(json.dumps({"event": "auto_merge", "task": task, "variant": variant}), flush=True)
        except Exception as exc:
            print(
                json.dumps(
                    {"event": "auto_merge_failed", "task": task, "variant": variant, "error": str(exc)}
                ),
                flush=True,
            )
            return False
        return path.is_file() and bool((read_json(path).get("progress") or {}).get("complete"))
    return False


def job_running(task: str, variant: str) -> bool:
    try:
        out = subprocess.check_output(["pgrep", "-af", "eval_per_seed.py|run_eval_group.py"], text=True)
    except subprocess.CalledProcessError:
        out = ""
    needle_task = f"--task {task}"
    needle_var = f"--variant {variant}"
    for line in out.splitlines():
        if needle_task in line and needle_var in line:
            return True
    lock = RUN / "results" / task / f".{variant}.launch.lock"
    if lock.is_file():
        try:
            pid = int(lock.read_text().strip().split()[0])
            os.kill(pid, 0)
            return True
        except (ValueError, OSError):
            # Stale lock only after confirming no workers; avoid launch storms.
            try:
                lock.unlink()
            except OSError:
                pass
            # Incomplete shards mean a prior group may still be finishing merger, or
            # crashed mid-run — prefer a brief cooldown via recent shard mtime.
            shard_dir = RUN / "results" / task
            if shard_dir.is_dir():
                recent = False
                now = time.time()
                for path in shard_dir.glob(f"{variant}_shard_*_of_*.json"):
                    try:
                        if now - path.stat().st_mtime < 180:
                            recent = True
                            break
                    except OSError:
                        pass
                if recent:
                    return True
    return False


def nvidia_busy() -> set[int]:
    uuid_to_idx = {}
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
    )
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            uuid_to_idx[parts[1]] = int(parts[0])
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
    )
    busy: set[int] = set()
    for line in apps.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        if parts[0] in uuid_to_idx:
            busy.add(uuid_to_idx[parts[0]])
    return busy


def traced_status() -> dict:
    if not STATE.is_file():
        return {"gate": False, "pending": 15, "running": 0, "completed": 0, "verify": False}
    s = read_json(STATE)
    counts = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
    for meta in (s.get("shards") or {}).values():
        st = meta.get("status") or "pending"
        counts[st] = counts.get(st, 0) + 1
    return {
        "gate": bool(s.get("place_eval_gate_passed")),
        "verify": bool(s.get("verify_done")),
        **counts,
    }


def place_merged() -> dict:
    out = {}
    for name in ("base200_confirm_easy", "base200_confirm_hard"):
        path = RUN / "results/place_container_plate" / f"{name}.json"
        out[name] = bool(path.is_file() and (read_json(path).get("progress") or {}).get("complete"))
    return out


def ensure_seeds(task: str, split: str, task_config: str) -> Path:
    out = RUN / "seeds" / f"{task}_{split}.json"
    if out.is_file():
        return out
    src = REPO / f"experiments/brace/seeds/multitask_v1/{task}.json"
    man = read_json(src)
    seeds = [int(x) for x in man["partitions"][split]]
    train = {int(x) for x in (REPO / f"data/{task}/demo_clean/seed.txt").read_text().split()}
    if set(seeds) & train:
        raise SystemExit(f"{task} {split} overlaps Base200 train seeds")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "task": task,
                "source_manifest": str(src.relative_to(REPO)),
                "source_manifest_status": man.get("status"),
                "eval_id": seeds,
                "train_rollout": [],
                "split": split,
                "task_config": task_config,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return out


def launch_eval(gpu: int, task: str, task_config: str, variant: str) -> None:
    split = "confirm_easy" if variant.endswith("easy") else "confirm_hard"
    seeds = ensure_seeds(task, split, task_config)
    seeds_arg = str(seeds.relative_to(REPO)) if seeds.is_absolute() else str(seeds)
    LOG.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log = LOG / f"{ts}_roll_gpu{gpu}_{task}_{variant}.log"
    # Keep ckpt paths repo-relative so resume matches earlier frozen-eval shards.
    ckpt = f"policy/DP/checkpoints/{task}-demo_clean-200-0/600.ckpt"
    cmd = [
        sys.executable,
        "experiments/brace/run_eval_group.py",
        "--workers",
        "3",
        "--",
        sys.executable,
        "experiments/phase1/eval_per_seed.py",
        "--task",
        task,
        "--task-config",
        task_config,
        "--variant",
        variant,
        "--ckpt-path",
        ckpt,
        "--seeds-file",
        seeds_arg,
        "--output-dir",
        "experiments/brace/runs/base200_frozen_eval_20260811T012027Z/results",
        "--id-repeats",
        "1",
        "--train-seed-count",
        "0",
        "--no-include-hard",
        "--resume",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = f"{REPO}:{REPO}/policy/DP" + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    with log.open("a", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            cmd, cwd=str(REPO), env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True
        )
    lock = RUN / "results" / task / f".{variant}.launch.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(f"{proc.pid} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "launch_eval",
                "gpu": gpu,
                "task": task,
                "variant": variant,
                "pid": proc.pid,
                "log": str(log),
            }
        ),
        flush=True,
    )


def main() -> int:
    os.chdir(REPO)
    seen_gate = False
    print(
        json.dumps(
            {
                "event": "watcher_start",
                "pool": POOL,
                "traced_pool": TRACED_POOL,
                "backlog_extra": BACKLOG_EXTRA,
            }
        ),
        flush=True,
    )
    while True:
        merged = place_merged()
        traced = traced_status()
        busy = nvidia_busy()
        print(
            json.dumps(
                {
                    "event": "tick",
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "place": merged,
                    "traced": traced,
                    "busy_gpus": sorted(busy),
                }
            ),
            flush=True,
        )
        if merged["base200_confirm_easy"] and merged["base200_confirm_hard"] and traced["gate"] and not seen_gate:
            print(json.dumps({"event": "place_eval_gate_passed_observed"}), flush=True)
            seen_gate = True

        traced_needs_gpus = traced["gate"] and int(traced.get("pending", 0)) > 0
        # While traced still needs cards, only schedule backlog on non-traced GPUs.
        candid_pool = BACKLOG_EXTRA if traced_needs_gpus else POOL
        free = [g for g in candid_pool if g not in busy]
        launched_this_tick: set[tuple[str, str]] = set()
        for gpu in free:
            launched = False
            for task, cfg, variant in BACKLOG:
                if (task, variant) in launched_this_tick:
                    continue
                if not ckpt_ready(task):
                    continue
                if job_complete(task, variant) or job_running(task, variant):
                    continue
                try:
                    launch_eval(gpu, task, cfg, variant)
                    launched_this_tick.add((task, variant))
                    launched = True
                    break
                except Exception as exc:
                    print(
                        json.dumps(
                            {
                                "event": "launch_failed",
                                "task": task,
                                "variant": variant,
                                "error": str(exc),
                            }
                        ),
                        flush=True,
                    )
            if not launched:
                break
            time.sleep(8)  # let CVD/nvidia register before next free check

        # Exit only when traced verified and no remaining backlog for ready ckpts.
        remaining = [
            (t, v)
            for t, _, v in BACKLOG
            if ckpt_ready(t) and not job_complete(t, v)
        ]
        if seen_gate and traced.get("verify") and not remaining:
            print(json.dumps({"event": "watcher_done"}), flush=True)
            return 0
        time.sleep(45)


if __name__ == "__main__":
    raise SystemExit(main())
