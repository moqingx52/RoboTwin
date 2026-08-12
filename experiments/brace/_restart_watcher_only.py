#!/usr/bin/env python3
import os
import signal
import subprocess
import time
from pathlib import Path

REPO = Path("/workspace/RoboTwin")
PY = "/root/miniconda/envs/RoboTwin/bin/python"
LOG = REPO / "experiments/brace/logs/base200_frozen_eval/watch_and_roll.log"
LOCK = (
    REPO
    / "experiments/brace/runs/base200_frozen_eval_20260811T012027Z/results/handover_mic"
    / ".base200_confirm_hard.launch.lock"
)


def iter_procs():
    for line in subprocess.check_output(["ps", "-eo", "pid,args"], text=True).splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        pid_s, _, args = line.partition(" ")
        try:
            yield int(pid_s), args
        except ValueError:
            continue


def main() -> None:
    os.chdir(REPO)
    killed = []
    for pid, args in iter_procs():
        if "_restart_watcher_only" in args:
            continue
        if "schedule_base200_place_traced" in args or "collect_traced_rollouts" in args:
            continue
        hit = False
        if "watch_and_roll_base200_pipeline.py" in args:
            hit = True
        if "--task handover_mic" in args and (
            "eval_per_seed.py" in args or "run_eval_group.py" in args
        ):
            hit = True
        if not hit:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except OSError as exc:
            print(f"kill_fail {pid} {exc}")
    print({"killed": killed})
    time.sleep(2)
    if LOCK.is_file():
        LOCK.unlink()
        print("cleared_lock")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO}:{REPO}/policy/DP" + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    with LOG.open("a", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            [PY, "experiments/brace/watch_and_roll_base200_pipeline.py"],
            cwd=str(REPO),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"WATCH={proc.pid}")
    time.sleep(55)
    print("tail:")
    for line in LOG.read_text(encoding="utf-8", errors="ignore").splitlines()[-15:]:
        print(line)
    print("handover:")
    for pid, args in iter_procs():
        if "handover_mic" in args and "eval" in args:
            print(pid, args[:160])
    print("sched:")
    for pid, args in iter_procs():
        if "schedule_base200_place_traced.py" in args:
            print(pid, args[:120])


if __name__ == "__main__":
    main()
