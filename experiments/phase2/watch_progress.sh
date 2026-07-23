#!/bin/bash
set -euo pipefail

state_path=${1:-experiments/phase2/run_state.json}
interval=${PHASE2_WATCH_INTERVAL:-10}

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required." >&2
  exit 1
fi

while true; do
  clear
  echo "============================================================"
  echo " Phase 2 progress"
  echo " Time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo " State: ${state_path}"
  echo "============================================================"
  if [[ ! -s "${state_path}" ]]; then
    echo "State file not created yet."
    sleep "${interval}"
    continue
  fi

  jq -r '
    .config.eval_per_gpu as $eval_per_gpu |
    "status: \(.status)",
    "updated: \(.updated_at)",
    "jobs: " + ((.job_counts // {}) | to_entries | map("\(.key)=\(.value)") | join("  ")),
    "",
    "GPU assignments:",
    (.gpu_assignments | to_entries[] |
      if .value.mode == "train" then
        "  GPU \(.key): TRAIN  \(.value.train)"
      elif .value.mode == "eval" then
        "  GPU \(.key): EVAL   \(.value.eval | length)/\($eval_per_gpu)  \(.value.eval | join(", "))"
      else
        "  GPU \(.key): IDLE"
      end),
    "",
    "Completed groups: \([.groups[] | select(.status == "completed")] | length)/\(.groups | length)",
    "",
    "Recent events:",
    (.events[-12:][]? | "  \(.time)  \(.message)")
  ' "${state_path}"

  echo
  echo "Ctrl+C stops only this monitor. Refresh: ${interval}s"
  sleep "${interval}"
done
