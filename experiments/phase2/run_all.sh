#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

read -r -a tasks <<< "${PHASE2_TASKS:-place_container_plate dump_bin_bigbin}"
read -r -a variants <<< "${PHASE2_VARIANTS:-expert_only uniform_mixed anchored_70 anchored_50 anchored_70_weighted}"
read -r -a train_seeds <<< "${PHASE2_TRAIN_SEEDS:-0}"
read -r -a gpu_ids <<< "${PHASE2_GPU_IDS:-0 1 2 3 4 5 6 7}"

exec python experiments/phase2/orchestrate.py \
  --tasks "${tasks[@]}" \
  --variants "${variants[@]}" \
  --train-seeds "${train_seeds[@]}" \
  --gpus "${gpu_ids[@]}" \
  --eval-per-gpu "${PHASE2_EVAL_PER_GPU:-3}" \
  --epochs "${PHASE2_EPOCHS:-50}" \
  --checkpoint-every "${PHASE2_CHECKPOINT_EVERY:-10}" \
  --learning-rate "${PHASE2_LR:-1e-5}" \
  --max-retries "${PHASE2_MAX_RETRIES:-3}" \
  --poll-interval "${PHASE2_POLL_INTERVAL:-5}" \
  --state-path "${PHASE2_STATE_PATH:-experiments/phase2/run_state.json}"
