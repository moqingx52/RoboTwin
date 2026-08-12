#!/usr/bin/env python3
"""Launch Base200 place branch on GPUs 0/1/2/4/5/6/7; leave GPU 3 alone."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

REPO = Path("/workspace/RoboTwin")
LOG = REPO / "experiments/brace/logs/traced_base200_v2/branch_place_base200_v2.log"
PY = "/root/miniconda/envs/RoboTwin/bin/python"


def main() -> None:
    os.chdir(REPO)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "BRACE_TASKS": "place_container_plate",
            "BRACE_TRACED_ROLLOUT_DIR": "experiments/brace/rollouts_traced_base200_v2",
            "BRACE_GPU_IDS": "0 1 2 4 5 6 7",
            "BRACE_AUDIT_WORKERS_PER_GPU": "3",
            "BRACE_AUDIT_WORKERS": "21",
            "BRACE_BRANCH_LABEL": "branches_place_base200_v2",
            "BRACE_PROTOCOL_V2_PATH": "experiments/brace/protocol.v2.3.json",
            "PYTHONPATH": f"{REPO}:{REPO}/policy/DP",
            "PATH": f"/root/miniconda/envs/RoboTwin/bin:{env.get('PATH','')}",
        }
    )
    # Ensure stack still exclusive on physical GPU 3 by never listing it.
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== branch start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===\n")
        handle.flush()
        proc = subprocess.Popen(
            ["bash", "experiments/brace/run_all.sh", "branch"],
            cwd=str(REPO),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"BRANCH_PID={proc.pid}")
    print(f"LOG={LOG}")
    time.sleep(8)
    # smoke: process alive + log grew
    print(f"alive={proc.poll() is None}")
    print("log_tail:")
    print("\n".join(LOG.read_text(encoding="utf-8", errors="ignore").splitlines()[-20:]))


if __name__ == "__main__":
    main()
