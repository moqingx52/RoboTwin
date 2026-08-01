#!/usr/bin/env bash
# Track B: place held-out confirmatory Stage-2 branch (separate from pilot archive).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

export BRACE_PROTOCOL_V2_PATH="${BRACE_PROTOCOL_V2_PATH:-experiments/brace/protocol.v2.3.json}"
export BRACE_TASKS=place_container_plate
export BRACE_BRANCH_LABEL=branches_confirm

brace_dir=experiments/brace
read -r -a tasks <<< "${BRACE_TASKS}"
# shellcheck source=experiments/brace/run_paths.sh
source "${repo_root}/experiments/brace/run_paths.sh"

exclude_file=experiments/brace/archive/branches_place_pilot_valid_v2.3/analyzed_seeds.json
if [[ ! -s "${exclude_file}" ]]; then
  echo "Missing analyzed seed exclusion list: ${exclude_file}" >&2
  exit 2
fi

BRACE_TRACED_ROLLOUT_DIR="${BRACE_TRACED_ROLLOUT_DIR:-experiments/brace/rollouts_traced}" \
  bash experiments/brace/run_all.sh select-confirm-seeds

export BRACE_PILOT_SEEDS_FILE
if ! BRACE_PILOT_SEEDS_FILE="$(brace_resolve_seeds_file place_container_plate_confirm_seeds.json)"; then
  echo "Missing confirm seeds; select-confirm-seeds failed." >&2
  exit 2
fi
export BRACE_TRACED_ROLLOUT_DIR=experiments/brace/rollouts_traced_pilot

if [[ ! -s experiments/brace/rollouts_traced_pilot/place_container_plate/manifest.jsonl ]]; then
  bash experiments/brace/run_all.sh collect-trace-pilot
  bash experiments/brace/run_all.sh verify-traced
fi

bash experiments/brace/run_all.sh branch

confirm_dir="$(brace_latest_branch_dir branches_confirm)"
pilot_summary="${brace_dir}/archive/branches_place_pilot_valid_v2.3/summary.json"
python experiments/brace/evaluate_confirmatory_gate.py \
  --pilot-summary "${pilot_summary}" \
  --confirm-summary "${confirm_dir}/summary.json" \
  --output "${confirm_dir}/merged_gate.json" || true

bash experiments/brace/archive_place_confirm.sh

echo "Place confirmatory branch complete: ${confirm_dir}/summary.json"
