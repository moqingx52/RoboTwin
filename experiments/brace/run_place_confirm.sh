#!/usr/bin/env bash
# Track B: place held-out confirmatory Stage-2 branch (separate from pilot archive).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

export BRACE_PROTOCOL_V2_PATH="${BRACE_PROTOCOL_V2_PATH:-experiments/brace/protocol.v2.3.json}"
export BRACE_TASKS=place_container_plate

exclude_file=experiments/brace/archive/branches_place_pilot_valid_v2.3/analyzed_seeds.json
if [[ ! -s "${exclude_file}" ]]; then
  echo "Missing analyzed seed exclusion list: ${exclude_file}" >&2
  exit 2
fi

BRACE_TRACED_ROLLOUT_DIR="${BRACE_TRACED_ROLLOUT_DIR:-experiments/brace/rollouts_traced}" \
  bash experiments/brace/run_all.sh select-confirm-seeds

export BRACE_PILOT_SEEDS_FILE=experiments/brace/seeds/place_container_plate_confirm_seeds.json
if [[ ! -s "${BRACE_PILOT_SEEDS_FILE}" ]]; then
  echo "Missing ${BRACE_PILOT_SEEDS_FILE}; select-confirm-seeds failed." >&2
  exit 2
fi
export BRACE_TRACED_ROLLOUT_DIR=experiments/brace/rollouts_traced_pilot
export BRACE_BRANCH_OUTPUT_DIR=experiments/brace/branches_confirm

if [[ ! -s experiments/brace/rollouts_traced_pilot/place_container_plate/manifest.jsonl ]]; then
  bash experiments/brace/run_all.sh collect-trace-pilot
  bash experiments/brace/run_all.sh verify-traced
fi

bash experiments/brace/run_all.sh branch

python experiments/brace/evaluate_confirmatory_gate.py \
  --pilot-summary experiments/brace/archive/branches_place_pilot_valid_v2.3/summary.json \
  --confirm-summary experiments/brace/branches_confirm/summary.json \
  --output experiments/brace/branches_confirm/merged_gate.json || true

bash experiments/brace/archive_place_confirm.sh

echo "Place confirmatory branch complete: experiments/brace/branches_confirm/summary.json"
