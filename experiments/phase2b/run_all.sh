#!/bin/bash
set -euo pipefail

stage=${1:?stage required: diagnose|screen|full_seed0|confirm_seeds}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

read -r -a tasks <<< "${PHASE2B_TASKS:-place_container_plate dump_bin_bigbin}"
read -r -a train_seeds <<< "${PHASE2B_TRAIN_SEEDS:-0}"
read -r -a confirm_seeds <<< "${PHASE2B_CONFIRM_SEEDS:-1 2}"
read -r -a gpu_ids <<< "${PHASE2B_GPU_IDS:-0 1 2 3 4 5 6}"
extra_args=()
if [[ "${PHASE2B_RETRY_FAILED:-0}" == "1" ]]; then
  extra_args+=(--retry-failed)
fi

state_default="experiments/phase2b/run_state_${stage}.json"
if [[ "${stage}" == "diagnose" ]]; then
  state_default="experiments/phase2b/run_state_diagnose.json"
fi

exec python experiments/phase2b/orchestrate.py "${stage}" \
  --tasks "${tasks[@]}" \
  --train-seeds "${train_seeds[@]}" \
  --confirm-seeds "${confirm_seeds[@]}" \
  --gpus "${gpu_ids[@]}" \
  --eval-per-gpu "${PHASE2B_EVAL_PER_GPU:-3}" \
  --epochs "${PHASE2B_EPOCHS:-20}" \
  --checkpoint-every "${PHASE2B_CHECKPOINT_EVERY:-5}" \
  --learning-rate "${PHASE2B_LR:-1e-5}" \
  --max-retries "${PHASE2B_MAX_RETRIES:-3}" \
  --retry-backoff "${PHASE2B_RETRY_BACKOFF:-60}" \
  --max-train-gpus "${PHASE2B_MAX_TRAIN_GPUS:-${#gpu_ids[@]}}" \
  --poll-interval "${PHASE2B_POLL_INTERVAL:-5}" \
  --full-top-k "${PHASE2B_FULL_TOP_K:-2}" \
  --state-path "${PHASE2B_STATE_PATH:-${state_default}}" \
  --screen-state-path "${PHASE2B_SCREEN_STATE_PATH:-experiments/phase2b/run_state_screen.json}" \
  --full-state-path "${PHASE2B_FULL_STATE_PATH:-experiments/phase2b/run_state_full_seed0.json}" \
  "${extra_args[@]}"
