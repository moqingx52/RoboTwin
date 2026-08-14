#!/bin/bash
# Wait for place Base200 traced verify_done, then run audit→branch→place pilot.
set -u
cd /workspace/RoboTwin || cd "$(dirname "$0")/../../.."
source /root/miniconda/etc/profile.d/conda.sh
conda activate RoboTwin
STATE=${BRACE_PLACE_TRACED_STATE:-experiments/brace/runs/base200_place_traced_15shard_20260811.state.json}
LOG=${BRACE_AFTER_TRACED_LOG:-experiments/brace/logs/traced_base200_v2/after_traced_trigger.log}
mkdir -p "$(dirname "$LOG")"
echo "start $(date -u +%FT%TZ)" | tee -a "$LOG"
while true; do
  if python - "$STATE" <<'PY'
import json, sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(
    "gate",
    s.get("place_eval_gate_passed"),
    "verify",
    s.get("verify_done"),
    "completed",
    sum(1 for m in s.get("shards", {}).values() if m.get("status") == "completed"),
    flush=True,
)
raise SystemExit(0 if s.get("verify_done") else 2)
PY
  then
    echo "TRIGGER_AFTER_TRACED $(date -u +%FT%TZ)" | tee -a "$LOG"
    export BRACE_GPU_IDS="${BRACE_GPU_IDS:-2 4 5 6 7}"
    export BRACE_TRACED_ROLLOUT_DIR="${BRACE_TRACED_ROLLOUT_DIR:-experiments/brace/rollouts_traced_base200_v2}"
    bash experiments/brace_v1/ops/run_place_line_a_after_traced.sh >>"$LOG" 2>&1
    echo "AFTER_TRACED_DONE $(date -u +%FT%TZ)" | tee -a "$LOG"
    exit 0
  fi
  sleep 120
done
