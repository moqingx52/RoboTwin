#!/usr/bin/env python3
"""Abort confirmed duplicate-audit orphan workers; annotate run; never touch stack train."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

REPO = Path("/workspace/RoboTwin")
DUP_RUN = REPO / "experiments/brace/runs/20260812T014347Z_audit_v2_place_container_plate"
SUPERSEDED = "experiments/brace/runs/20260811T214147Z_audit_v2_place_container_plate"
ORPHAN_LO = 2252389
ORPHAN_HI = 2252408
STACK_NEEDLE = "task.name=stack_bowls_three"


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def stack_pids() -> list[int]:
    out = []
    for line in subprocess.check_output(["ps", "-eo", "pid,args"], text=True).splitlines():
        if STACK_NEEDLE in line and "train.py" in line:
            out.append(int(line.strip().split(None, 1)[0]))
    return out


def main() -> None:
    os.chdir(REPO)
    stack_before = stack_pids()
    if not stack_before:
        raise SystemExit("REFUSE: stack_bowls_three train.py not found; abort cleanup")
    print(json.dumps({"event": "stack_protected", "pids": stack_before}))

    orphans = []
    for pid in range(ORPHAN_LO, ORPHAN_HI + 1):
        if not alive(pid):
            continue
        cmd = cmdline(pid)
        if "multiprocessing.spawn" not in cmd:
            print(json.dumps({"event": "skip_non_orphan", "pid": pid, "cmd": cmd[:120]}))
            continue
        orphans.append(pid)
    print(json.dumps({"event": "orphans_confirmed", "pids": orphans, "n": len(orphans)}))

    # Annotate duplicate run (do not delete).
    note = {
        "status": "operational_duplicate_aborted",
        "superseded_by": SUPERSEDED,
        "reason": (
            "Redundant audit launched after already-passing "
            "20260811T214147Z; parent aborted; orphan spawn workers cleaned."
        ),
        "annotated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "orphan_pids": orphans,
        "orphan_start_utc": "2026-08-12T01:43:47Z",
        "log": "experiments/brace/logs/traced_base200_v2/after_traced_continue.log",
        "preserve_artifacts": True,
    }
    DUP_RUN.mkdir(parents=True, exist_ok=True)
    (DUP_RUN / "operational_status.json").write_text(json.dumps(note, indent=2) + "\n")
    meta_path = DUP_RUN / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        meta["operational_status"] = note["status"]
        meta["superseded_by"] = SUPERSEDED
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"event": "annotated", "path": str(DUP_RUN / "operational_status.json")}))

    # SIGINT first
    for pid in orphans:
        try:
            os.kill(pid, signal.SIGINT)
            print(json.dumps({"event": "sigint", "pid": pid}))
        except OSError as exc:
            print(json.dumps({"event": "sigint_fail", "pid": pid, "error": str(exc)}))
    time.sleep(8)
    survivors = [pid for pid in orphans if alive(pid)]
    print(json.dumps({"event": "after_sigint", "survivors": survivors}))
    for pid in survivors:
        try:
            os.kill(pid, signal.SIGKILL)
            print(json.dumps({"event": "sigkill", "pid": pid}))
        except OSError as exc:
            print(json.dumps({"event": "sigkill_fail", "pid": pid, "error": str(exc)}))
    time.sleep(2)
    still = [pid for pid in orphans if alive(pid)]
    stack_after = stack_pids()
    print(
        json.dumps(
            {
                "event": "cleanup_done",
                "still_alive": still,
                "stack_before": stack_before,
                "stack_after": stack_after,
                "stack_ok": bool(set(stack_before) & set(stack_after)) or bool(stack_after),
            }
        )
    )
    if still:
        raise SystemExit(f"orphans still alive: {still}")
    if not stack_after:
        raise SystemExit("FATAL: stack train disappeared during cleanup")


if __name__ == "__main__":
    main()
