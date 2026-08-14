#!/usr/bin/env python3
"""Kill the SIGSTOP'd preservation scheduler after tail reshard marks jobs complete."""
from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

STATE = Path(
    "/workspace/RoboTwin/experiments/brace/runs/20260812T031200Z_place_base200_v2_line_a_pilot/preservation_eval_state.json"
)
NEED = {"pres:B3:seed1", "pres:B3:seed3", "pres:B3:seed4", "pres:B3:seed5"}
SCHED = 794152


def main() -> int:
    print("waiting for preservation jobs to be marked completed", flush=True)
    while True:
        state = json.loads(STATE.read_text(encoding="utf-8"))
        done = {
            j["id"]
            for j in state.get("jobs") or []
            if j.get("id") in NEED and j.get("status") == "completed"
        }
        counts: dict[str, int] = {}
        for job in state.get("jobs") or []:
            status = str(job.get("status"))
            counts[status] = counts.get(status, 0) + 1
        print(time.strftime("%H:%M:%S"), "done", sorted(done), "counts", counts, flush=True)
        if done == NEED:
            break
        time.sleep(20)
    if Path(f"/proc/{SCHED}").exists():
        os.kill(SCHED, signal.SIGKILL)
        print(f"SIGKILL stopped scheduler {SCHED}", flush=True)
    else:
        print("scheduler already gone", flush=True)
    print("release done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
