#!/bin/bash
set -euo pipefail

stage=${1:-all}
num_gpus=${2:-8}
tasks=(move_can_pot place_container_plate click_alarmclock dump_bin_bigbin)
variants=(success seed_balanced difficulty_weighted)

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
phase_dir="${repo_root}/experiments/phase1"

run_base_all() {
  for i in "${!tasks[@]}"; do
    task="${tasks[$i]}"
    gpu_id=$((i % num_gpus))
    bash "${phase_dir}/run_task.sh" "${task}" "${gpu_id}" base &
  done
  wait
}

run_seeds_all() {
  for task in "${tasks[@]}"; do
    bash "${phase_dir}/run_task.sh" "${task}" 0 seeds &
  done
  wait
}

run_rollout_all() {
  # 4 tasks x 2 shards = 8 independent rollout workers on an 8-GPU node.
  local gpu_id=0
  local shards_per_task=2
  for task in "${tasks[@]}"; do
    for shard_id in $(seq 0 $((shards_per_task - 1))); do
      bash "${phase_dir}/run_task.sh" "${task}" "${gpu_id}" rollout "${shard_id}" "${shards_per_task}" &
      gpu_id=$(((gpu_id + 1) % num_gpus))
    done
  done
  wait

  for task in "${tasks[@]}"; do
    bash "${phase_dir}/run_task.sh" "${task}" 0 merge_rollout &
  done
  wait
}

run_build_all() {
  for task in "${tasks[@]}"; do
    bash "${phase_dir}/run_task.sh" "${task}" 0 build &
  done
  wait
}

run_finetune_all() {
  local job_count=0
  for task in "${tasks[@]}"; do
    for variant in "${variants[@]}"; do
      gpu_id=$((job_count % num_gpus))
      bash "${phase_dir}/finetune.sh" "${task}" "${variant}" "${gpu_id}" 0 200 14 &
      job_count=$((job_count + 1))
      if (( job_count % num_gpus == 0 )); then
        wait
      fi
    done
  done
  wait
}

run_eval_all() {
  local job_count=0
  for task in "${tasks[@]}"; do
    gpu_id=$((job_count % num_gpus))
    (
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      cd "${repo_root}"
      python "${phase_dir}/eval_per_seed.py" --task "${task}" --task-config demo_clean --variant base \
        --ckpt-path "${repo_root}/policy/DP/checkpoints/${task}-demo_clean-50-0/600.ckpt"
    ) &
    job_count=$((job_count + 1))
    if (( job_count % num_gpus == 0 )); then
      wait
    fi

    for variant in "${variants[@]}"; do
      gpu_id=$((job_count % num_gpus))
      (
        export CUDA_VISIBLE_DEVICES="${gpu_id}"
        cd "${repo_root}"
        python "${phase_dir}/eval_per_seed.py" --task "${task}" --task-config demo_clean --variant "${variant}" \
          --ckpt-path "${repo_root}/policy/DP/checkpoints/${task}-${variant}-0/200.ckpt"
      ) &
      job_count=$((job_count + 1))
      if (( job_count % num_gpus == 0 )); then
        wait
      fi
    done
  done
  wait

  cd "${repo_root}"
  python "${phase_dir}/plot_diagnostics.py"
}

case "${stage}" in
  base) run_base_all ;;
  seeds) run_seeds_all ;;
  rollout) run_rollout_all ;;
  build) run_build_all ;;
  finetune) run_finetune_all ;;
  eval) run_eval_all ;;
  all)
    run_base_all
    run_seeds_all
    run_rollout_all
    run_build_all
    run_finetune_all
    run_eval_all
    ;;
  *)
    echo "stage must be one of: base, seeds, rollout, build, finetune, eval, all" >&2
    exit 1
    ;;
esac

