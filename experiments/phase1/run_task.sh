#!/bin/bash
set -euo pipefail

task_name=${1}
gpu_id=${2:-0}
stage=${3:-all}
shard_id=${4:-0}
num_shards=${5:-1}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
phase_dir="${repo_root}/experiments/phase1"

run_base() {
  cd "${repo_root}"
  bash collect_data.sh "${task_name}" demo_clean "${gpu_id}"
  cd "${repo_root}/policy/DP"
  bash process_data.sh "${task_name}" demo_clean 50
  bash train.sh "${task_name}" demo_clean 50 0 14 "${gpu_id}"
}

run_seed_split() {
  cd "${repo_root}"
  python "${phase_dir}/generate_seed_splits.py" --task "${task_name}" --task-config demo_clean
}

run_rollout() {
  cd "${repo_root}"
  export CUDA_VISIBLE_DEVICES="${gpu_id}"
  python "${phase_dir}/collect_rollouts.py" \
    --task "${task_name}" \
    --task-config demo_clean \
    --rollouts-per-seed 8 \
    --train-seed 0 \
    --checkpoint-num 600 \
    --action-dim 14 \
    --shard-id "${shard_id}" \
    --num-shards "${num_shards}" \
    --resume
}

run_merge_rollout() {
  cd "${repo_root}"
  python "${phase_dir}/merge_rollout_shards.py" \
    --task "${task_name}" \
    --task-config demo_clean \
    --rollouts-per-seed 8
}

run_build() {
  cd "${repo_root}"
  for variant in success seed_balanced difficulty_weighted; do
    python "${phase_dir}/build_dataset.py" --task "${task_name}" --task-config demo_clean --variant "${variant}"
  done
}

run_finetune() {
  cd "${repo_root}"
  for variant in success seed_balanced difficulty_weighted; do
    bash "${phase_dir}/finetune.sh" "${task_name}" "${variant}" "${gpu_id}" 0 200 14
  done
}

run_eval() {
  cd "${repo_root}"
  python "${phase_dir}/eval_per_seed.py" --task "${task_name}" --task-config demo_clean --variant base \
    --ckpt-path "${repo_root}/policy/DP/checkpoints/${task_name}-demo_clean-50-0/600.ckpt"
  for variant in success seed_balanced difficulty_weighted; do
    python "${phase_dir}/eval_per_seed.py" --task "${task_name}" --task-config demo_clean --variant "${variant}" \
      --ckpt-path "${repo_root}/policy/DP/checkpoints/${task_name}-${variant}-0/200.ckpt"
  done
}

case "${stage}" in
  base) run_base ;;
  seeds) run_seed_split ;;
  rollout) run_rollout ;;
  merge_rollout) run_merge_rollout ;;
  build) run_build ;;
  finetune) run_finetune ;;
  eval) run_eval ;;
  all)
    run_base
    run_seed_split
    run_rollout
    run_build
    run_finetune
    run_eval
    ;;
  *)
    echo "stage must be one of: base, seeds, rollout, merge_rollout, build, finetune, eval, all" >&2
    exit 1
    ;;
esac

