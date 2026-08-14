#!/usr/bin/env python3
"""Restart traced scheduler + backlog watcher + after-traced trigger."""
from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

REPO = Path("/workspace/RoboTwin")
LOG_SCHED = REPO / "experiments/brace/logs/traced_base200_v2/scheduler.log"
LOG_WATCH = REPO / "experiments/brace/logs/base200_frozen_eval/watch_and_roll.log"
LOG_TRIG = REPO / "experiments/brace/logs/traced_base200_v2/after_traced_trigger.log"
STATE_LOCK = REPO / "experiments/brace/runs/base200_place_traced_15shard_20260811.state.lock"
PY = "/root/miniconda/envs/RoboTwin/bin/python"


def pids_for(needle: str) -> list[int]:
    out: list[int] = []
    try:
        text = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    except subprocess.CalledProcessError:
        return out
    for line in text.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        pid_s, _, args = line.partition(" ")
        if needle in args and "pids_for" not in args and "_restart_line_a" not in args:
            try:
                out.append(int(pid_s))
            except ValueError:
                pass
    return out


def kill_needles(needles: list[str]) -> None:
    for needle in needles:
        for pid in pids_for(needle):
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"killed {needle} pid={pid}")
            except OSError as exc:
                print(f"kill_fail {pid} {exc}")


def main() -> None:
    os.chdir(REPO)
    kill_needles(
        [
            "experiments/brace_v1/ops/schedule_base200_place_traced.py",
            "experiments/brace_v1/ops/watch_and_roll_base200_pipeline.py",
        ]
    )
    time.sleep(2)
    if STATE_LOCK.is_file():
        STATE_LOCK.unlink()
        print("cleared_lock")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO}:{REPO}/policy/DP" + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )

    LOG_SCHED.parent.mkdir(parents=True, exist_ok=True)
    LOG_WATCH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_SCHED.open("a", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            [
                PY,
                "experiments/brace_v1/ops/schedule_base200_place_traced.py",
                "--eval-run-dir",
                "experiments/brace/runs/base200_frozen_eval_20260811T012027Z",
                "--state",
                "experiments/brace/runs/base200_place_traced_15shard_20260811.state.json",
                "--rollout-dir",
                "experiments/brace/rollouts_traced_base200_v2",
                "--gpus",
                "2",
                "4",
                "5",
                "6",
                "7",
                "--num-shards",
                "15",
                "--workers-per-gpu",
                "3",
                "--poll-seconds",
                "45",
            ],
            cwd=str(REPO),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"SCHED={proc.pid}")

    with LOG_WATCH.open("a", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            [PY, "experiments/brace_v1/ops/watch_and_roll_base200_pipeline.py"],
            cwd=str(REPO),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"WATCH={proc.pid}")

    if not pids_for("trigger_place_line_a_after_traced.sh"):
        with LOG_TRIG.open("a", encoding="utf-8") as handle:
            proc = subprocess.Popen(
                ["bash", "experiments/brace_v1/ops/trigger_place_line_a_after_traced.sh"],
                cwd=str(REPO),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(f"TRIG={proc.pid}")
    else:
        print("TRIG=already_running")

    time.sleep(6)
    print("collect:")
    for line in subprocess.check_output(["ps", "-eo", "pid,args"], text=True).splitlines():
        if "collect_traced_rollouts.py" in line and "place_container_plate" in line:
            print(line.strip()[:200])
    print("controllers:")
    for needle in (
        "schedule_base200_place_traced.py",
        "watch_and_roll_base200_pipeline.py",
        "trigger_place_line_a_after_traced.sh",
    ):
        print(needle, pids_for(needle))
    # show last scheduler lines
    if LOG_SCHED.is_file():
        lines = LOG_SCHED.read_text(encoding="utf-8", errors="ignore").splitlines()
        print("sched_tail:")
        for line in lines[-12:]:
            print(line)


if __name__ == "__main__":
    main()
