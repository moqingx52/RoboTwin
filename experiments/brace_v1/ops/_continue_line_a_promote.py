#!/usr/bin/env python3
import os
import signal
import subprocess
import time
from pathlib import Path

REPO = Path("/workspace/RoboTwin")


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
    for pid, args in iter_procs():
        if "_continue_line_a" in args:
            continue
        hit = False
        if "replay_audit_v2.py" in args:
            hit = True
        if "run_place_line_a_after_traced.sh" in args:
            hit = True
        if not hit:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            print("killed", pid)
        except OSError as exc:
            print("fail", pid, exc)
    time.sleep(2)

    audit_root = Path((REPO / "experiments/brace/runs/LATEST_AUDIT_place_container_plate").read_text().strip())
    # Prefer the known-good Aug11 audit if LATEST was overwritten by a partial re-run.
    preferred = REPO / "experiments/brace/runs/20260811T214147Z_audit_v2_place_container_plate"
    if preferred.is_dir():
        summary = preferred / "replay_audit_v2/place_container_plate/summary.json"
        if summary.is_file():
            audit_root = preferred
            (REPO / "experiments/brace/runs/LATEST_AUDIT_place_container_plate").write_text(
                str(preferred.relative_to(REPO)) + "\n"
            )
            print("restored_LATEST", preferred)
    promote_run = audit_root / "replay_audit_v2/place_container_plate"
    if not promote_run.is_absolute():
        promote_run = REPO / promote_run
    print("promote_run", promote_run)
    env = os.environ.copy()
    env["BRACE_PROMOTE_RUN"] = str(promote_run.relative_to(REPO))
    env["BRACE_PROMOTE_TARGET"] = "archive/replay_audit_v2_place_base200_v2_gate"
    env["PYTHONPATH"] = f"{REPO}:{REPO}/policy/DP"
    subprocess.check_call(
        ["bash", "experiments/brace/run_all.sh", "promote-run"],
        cwd=str(REPO),
        env=env,
    )
    print("PROMOTE_OK")


if __name__ == "__main__":
    main()
