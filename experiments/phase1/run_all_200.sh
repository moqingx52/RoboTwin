#!/bin/bash
set -euo pipefail

stage=${1:-build}
num_gpus=${2:-8}

read -r -a tasks <<< "${TASKS:-move_can_pot place_container_plate click_alarmclock dump_bin_bigbin}"
read -r -a train_seeds <<< "${TRAIN_SEEDS:-0}"
read -r -a variants <<< "${VARIANTS:-expert_only success seed_balanced difficulty_weighted}"

expert_data_num=200
base_train_seed=${BASE_TRAIN_SEED:-0}
checkpoint_num=${CHECKPOINT_NUM:-600}
epochs=${FINETUNE_EPOCHS:-200}
steps_per_epoch_setting=${FINETUNE_STEPS_PER_EPOCH:-auto}
rollouts_per_seed=${ROLLOUTS_PER_SEED:-8}
shards_per_task=${SHARDS_PER_TASK:-2}
eval_shards=${EVAL_SHARDS_PER_JOB:-8}
action_dim=${ACTION_DIM:-14}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
phase_dir="${repo_root}/experiments/phase1"
rollout_dir="${phase_dir}/rollouts_200"
data_dir="${repo_root}/policy/DP/data_phase1_200"
eval_dir="${phase_dir}/eval_results_200"
figure_dir="${phase_dir}/figures_200"
log_dir="${phase_dir}/logs_200"
mkdir -p "${log_dir}"

pids=()
wait_jobs() {
  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  pids=()
  if (( failed != 0 )); then
    echo "One or more jobs failed. Inspect ${log_dir}." >&2
    return 1
  fi
}

# Keep up to num_gpus workers busy; assign a free GPU as soon as any job finishes.
run_gpu_pool() {
  local -a jobs=("$@")
  local failed=0
  local job_idx=0
  local running=0
  local total=${#jobs[@]}
  local -a slot_pids=()
  local g

  for ((g=0; g<num_gpus; g++)); do
    slot_pids[g]=0
  done

  while (( job_idx < total || running > 0 )); do
    for ((g=0; g<num_gpus; g++)); do
      if (( slot_pids[g] != 0 )) && ! kill -0 "${slot_pids[g]}" 2>/dev/null; then
        wait "${slot_pids[g]}" || failed=1
        slot_pids[g]=0
        running=$((running - 1))
      fi
    done
    for ((g=0; g<num_gpus; g++)); do
      if (( job_idx < total && slot_pids[g] == 0 )); then
        CUDA_VISIBLE_DEVICES="${g}" bash -lc "${jobs[job_idx]}" &
        slot_pids[g]=$!
        job_idx=$((job_idx + 1))
        running=$((running + 1))
      fi
    done
    if (( running > 0 )); then
      sleep 1
    fi
  done

  if (( failed != 0 )); then
    echo "One or more pooled jobs failed. Inspect ${log_dir}." >&2
    return 1
  fi
}

run_eval_sharded() {
  local log_prefix=$1
  local task=$2
  local variant=$3
  local output_dir=$4
  shift 4
  local -a extra_args=("$@")
  local shard

  pids=()
  for ((shard=0; shard<eval_shards; shard++)); do
    local gpu=$((shard % num_gpus))
    (
      cd "${repo_root}"
      export CUDA_VISIBLE_DEVICES="${gpu}"
      python "${phase_dir}/eval_per_seed.py" \
        --task "${task}" --task-config demo_clean --variant "${variant}" \
        --output-dir "${output_dir}" \
        --shard-id "${shard}" --num-shards "${eval_shards}" \
        "${extra_args[@]}"
    ) >"${log_dir}/${log_prefix}_shard${shard}.log" 2>&1 &
    pids+=("$!")
  done
  wait_jobs

  cd "${repo_root}"
  python "${phase_dir}/merge_eval_shards.py" \
    --task "${task}" --task-config demo_clean --variant "${variant}" \
    --output-dir "${output_dir}" --num-shards "${eval_shards}"
}

run_rollout() {
  local -a pool_jobs=()
  local task shard
  for task in "${tasks[@]}"; do
    for ((shard=0; shard<shards_per_task; shard++)); do
      pool_jobs+=(
        "cd \"${repo_root}\" && python \"${phase_dir}/collect_rollouts.py\" \
          --task \"${task}\" --task-config demo_clean \
          --expert-data-num \"${expert_data_num}\" \
          --train-seed \"${base_train_seed}\" --checkpoint-num \"${checkpoint_num}\" \
          --rollouts-per-seed \"${rollouts_per_seed}\" --action-dim \"${action_dim}\" \
          --shard-id \"${shard}\" --num-shards \"${shards_per_task}\" --resume \
          --output-dir \"${rollout_dir}\" \
          >\"${log_dir}/rollout_${task}_shard${shard}.log\" 2>&1"
      )
    done
  done
  run_gpu_pool "${pool_jobs[@]}"
}

run_verify() {
  cd "${repo_root}"
  python "${phase_dir}/verify_rollouts.py" \
    --tasks "${tasks[@]}" --rollout-dir "${rollout_dir}" \
    --rollouts-per-seed "${rollouts_per_seed}" --num-shards "${shards_per_task}"
}

run_merge() {
  local task
  run_verify
  cd "${repo_root}"
  for task in "${tasks[@]}"; do
    python "${phase_dir}/merge_rollout_shards.py" \
      --task "${task}" --task-config demo_clean \
      --rollouts-per-seed "${rollouts_per_seed}" --rollout-dir "${rollout_dir}"
  done
}

run_build() {
  local task variant
  mkdir -p "${data_dir}"
  cd "${repo_root}"
  # Dataset conversion is intentionally serial: parallel image concatenation
  # creates large RAM and disk-I/O spikes.
  for task in "${tasks[@]}"; do
    for variant in "${variants[@]}"; do
      python "${phase_dir}/build_dataset.py" \
        --task "${task}" --task-config demo_clean --variant "${variant}" \
        --expert-data-num "${expert_data_num}" \
        --rollout-dir "${rollout_dir}" --output-dir "${data_dir}"
    done
  done
}

run_finetune() {
  local -a pool_jobs=()
  local task variant seed task_steps_per_epoch
  for task in "${tasks[@]}"; do
    if [[ "${steps_per_epoch_setting}" == "auto" ]]; then
      task_steps_per_epoch=$(python "${phase_dir}/dataset_train_batches.py" \
        "${data_dir}/${task}-expert_only.zarr" --batch-size 128)
    else
      task_steps_per_epoch="${steps_per_epoch_setting}"
    fi
    echo "${task}: fixed training batches per epoch=${task_steps_per_epoch} (expert-only reference)"
    for variant in "${variants[@]}"; do
      for seed in "${train_seeds[@]}"; do
        pool_jobs+=(
          "cd \"${repo_root}\" && bash \"${phase_dir}/finetune.sh\" \
            \"${task}\" \"${variant}\" \"\${CUDA_VISIBLE_DEVICES}\" \"${seed}\" \"${epochs}\" \"${action_dim}\" \
            \"${expert_data_num}\" \"data_phase1_200\" \"${base_train_seed}\" \"${task_steps_per_epoch}\" \
            >\"${log_dir}/finetune_${task}_${variant}_seed${seed}.log\" 2>&1"
        )
      done
    done
  done
  run_gpu_pool "${pool_jobs[@]}"
}

run_eval() {
  local task variant seed seed_eval_dir hard_seeds_file

  # Rank difficulty only on held-out eval seeds. Eight repeats reduce the
  # discretization noise before choosing the shared hard-20 split.
  local probe_dir="${eval_dir}/base_probe"
  local hard_dir="${eval_dir}/hard_eval_seeds"
  mkdir -p "${probe_dir}" "${hard_dir}"
  for task in "${tasks[@]}"; do
    run_eval_sharded "eval_probe_${task}_base" "${task}" base "${probe_dir}" \
      --ckpt-path "${repo_root}/policy/DP/checkpoints/${task}-demo_clean-200-${base_train_seed}/${checkpoint_num}.ckpt" \
      --rollout-dir "${rollout_dir}" \
      --id-repeats 8 --train-repeats 0 --hard-repeats 0
  done

  cd "${repo_root}"
  for task in "${tasks[@]}"; do
    python "${phase_dir}/select_hard_eval_seeds.py" \
      --base-eval "${probe_dir}/${task}/base.json" --count 20 \
      --output "${hard_dir}/${task}.json"
  done

  for seed in "${train_seeds[@]}"; do
    seed_eval_dir="${eval_dir}/train_seed_${seed}"
    mkdir -p "${seed_eval_dir}"
    for task in "${tasks[@]}"; do
      hard_seeds_file="${hard_dir}/${task}.json"
      run_eval_sharded "eval_${task}_base_seed${seed}" "${task}" base "${seed_eval_dir}" \
        --ckpt-path "${repo_root}/policy/DP/checkpoints/${task}-demo_clean-200-${base_train_seed}/${checkpoint_num}.ckpt" \
        --rollout-dir "${rollout_dir}" \
        --hard-seeds-file "${hard_seeds_file}" --policy-seed-offset 1000

      for variant in "${variants[@]}"; do
        run_eval_sharded "eval_${task}_${variant}_seed${seed}" "${task}" "${variant}" "${seed_eval_dir}" \
          --ckpt-path "${repo_root}/policy/DP/checkpoints/${task}-${variant}-${seed}/${epochs}.ckpt" \
          --rollout-dir "${rollout_dir}" \
          --hard-seeds-file "${hard_seeds_file}" --policy-seed-offset 1000
      done
    done

    cd "${repo_root}"
    python "${phase_dir}/plot_diagnostics.py" \
      --tasks "${tasks[@]}" --variants base "${variants[@]}" --rollout-dir "${rollout_dir}" \
      --eval-dir "${seed_eval_dir}" --output-dir "${figure_dir}/train_seed_${seed}"
  done

  cd "${repo_root}"
  python "${phase_dir}/aggregate_eval.py" \
    --eval-dir "${eval_dir}" --train-seeds "${train_seeds[@]}" --tasks "${tasks[@]}" \
    --variants base "${variants[@]}" \
    --output "${eval_dir}/summary.json"
}

case "${stage}" in
  rollout) run_rollout; run_merge ;;
  verify) run_verify ;;
  merge) run_merge ;;
  build) run_build ;;
  finetune) run_finetune ;;
  eval) run_eval ;;
  all) run_rollout; run_merge; run_build; run_finetune; run_eval ;;
  *)
    echo "stage must be one of: rollout, verify, merge, build, finetune, eval, all" >&2
    exit 1
    ;;
esac
