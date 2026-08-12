#!/bin/bash
# Lightweight monitor for place_base200_v2 pilot (GPU + job state).
set -uo pipefail
REPO=/workspace/RoboTwin
RUN="$REPO/experiments/brace/runs/20260812T031200Z_place_base200_v2_line_a_pilot"
LOG="$RUN/logs/watch_pilot.log"
STATE="$RUN/train_state.json"
mkdir -p "$RUN/logs"
while true; do
  {
    echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    /root/miniconda/envs/RoboTwin/bin/python - <<'PY'
import json
from pathlib import Path
from collections import Counter
s=json.load(open("/workspace/RoboTwin/experiments/brace/runs/20260812T031200Z_place_base200_v2_line_a_pilot/train_state.json"))
print("jobs", dict(Counter(j["status"] for j in s["jobs"])))
for j in s["jobs"]:
    if j["status"]=="running":
        pid=j.get("pid")
        alive=bool(pid) and Path(f"/proc/{pid}").exists()
        print(f"  {j['id']} gpu={j.get('gpu')} pid={pid} alive={alive}")
pending=[j["id"] for j in s["jobs"] if j["status"]=="pending"]
if pending:
    print("pending", pending)
sched=Path("/proc").glob("*")
PY
    pgrep -af "_schedule_place_base200_v2_pilot.py" | grep -v grep | head -1 || echo "scheduler: DOWN"
    df -h /workspace | tail -1
  } >>"$LOG" 2>&1
  sleep 60
done
