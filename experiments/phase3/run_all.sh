#!/bin/bash
set -euo pipefail

stage=${1:?stage required: init|prep|screen|full|iteration}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

read -r -a tasks <<< "${PHASE3_TASKS:-place_container_plate dump_bin_bigbin}"
read -r -a gpu_ids <<< "${PHASE3_GPU_IDS:-0 1 2 3 4 5 6}"

state_default="experiments/phase3/run_state_${stage}.json"

exec python experiments/phase3/orchestrate.py "${stage}" \
  --tasks "${tasks[@]}" \
  --gpus "${gpu_ids[@]}" \
  --state-path "${PHASE3_STATE_PATH:-${state_default}}"
