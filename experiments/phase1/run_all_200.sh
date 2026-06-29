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

run_rollout() {
  local job=0
  local task shard gpu
  for task in "${tasks[@]}"; do
    for ((shard=0; shard<shards_per_task; shard++)); do
      gpu=$((job % num_gpus))
      (
        cd "${repo_root}"
        export CUDA_VISIBLE_DEVICES="${gpu}"
        python "${phase_dir}/collect_rollouts.py" \
          --task "${task}" --task-config demo_clean \
          --expert-data-num "${expert_data_num}" \
          --train-seed "${base_train_seed}" --checkpoint-num "${checkpoint_num}" \
          --rollouts-per-seed "${rollouts_per_seed}" --action-dim "${action_dim}" \
          --shard-id "${shard}" --num-shards "${shards_per_task}" --resume \
          --output-dir "${rollout_dir}"
      ) >"${log_dir}/rollout_${task}_shard${shard}.log" 2>&1 &
      pids+=("$!")
      job=$((job + 1))
      if (( ${#pids[@]} == num_gpus )); then wait_jobs; fi
    done
  done
  if (( ${#pids[@]} > 0 )); then wait_jobs; fi
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
  local job=0
  local task variant seed gpu task_steps_per_epoch
  for task in "${tasks[@]}"; do
    if [[ "${steps_per_epoch_setting}" == "auto" ]]; then
      task_steps_per_epoch=$(python "${phase_dir}/dataset_train_batches.py" \
        "${data_dir}/${task}-success.zarr" --batch-size 128)
    else
      task_steps_per_epoch="${steps_per_epoch_setting}"
    fi
    echo "${task}: fixed training batches per epoch=${task_steps_per_epoch}"
    for variant in "${variants[@]}"; do
      for seed in "${train_seeds[@]}"; do
        gpu=$((job % num_gpus))
        bash "${phase_dir}/finetune.sh" \
          "${task}" "${variant}" "${gpu}" "${seed}" "${epochs}" "${action_dim}" \
          "${expert_data_num}" "data_phase1_200" "${base_train_seed}" "${task_steps_per_epoch}" \
          >"${log_dir}/finetune_${task}_${variant}_seed${seed}.log" 2>&1 &
        pids+=("$!")
        job=$((job + 1))
        if (( ${#pids[@]} == num_gpus )); then wait_jobs; fi
      done
    done
  done
  if (( ${#pids[@]} > 0 )); then wait_jobs; fi
}

run_eval() {
  local job=0
  local task variant seed gpu seed_eval_dir hard_seeds_file

  # Rank difficulty only on held-out eval seeds. Eight repeats reduce the
  # discretization noise before choosing the shared hard-20 split.
  local probe_dir="${eval_dir}/base_probe"
  local hard_dir="${eval_dir}/hard_eval_seeds"
  mkdir -p "${probe_dir}" "${hard_dir}"
  for task in "${tasks[@]}"; do
    gpu=$((job % num_gpus))
    (
      cd "${repo_root}"
      export CUDA_VISIBLE_DEVICES="${gpu}"
      python "${phase_dir}/eval_per_seed.py" \
        --task "${task}" --task-config demo_clean --variant base \
        --ckpt-path "${repo_root}/policy/DP/checkpoints/${task}-demo_clean-200-${base_train_seed}/${checkpoint_num}.ckpt" \
        --rollout-dir "${rollout_dir}" --output-dir "${probe_dir}" \
        --id-repeats 8 --train-repeats 0 --hard-repeats 0
    ) >"${log_dir}/eval_probe_${task}_base.log" 2>&1 &
    pids+=("$!")
    job=$((job + 1))
    if (( ${#pids[@]} == num_gpus )); then wait_jobs; fi
  done
  if (( ${#pids[@]} > 0 )); then wait_jobs; fi

  cd "${repo_root}"
  for task in "${tasks[@]}"; do
    python "${phase_dir}/select_hard_eval_seeds.py" \
      --base-eval "${probe_dir}/${task}/base.json" --count 20 \
      --output "${hard_dir}/${task}.json"
  done

  for seed in "${train_seeds[@]}"; do
    seed_eval_dir="${eval_dir}/train_seed_${seed}"
    mkdir -p "${seed_eval_dir}"
    job=0
    for task in "${tasks[@]}"; do
      hard_seeds_file="${hard_dir}/${task}.json"
      gpu=$((job % num_gpus))
      (
        cd "${repo_root}"
        export CUDA_VISIBLE_DEVICES="${gpu}"
        python "${phase_dir}/eval_per_seed.py" \
          --task "${task}" --task-config demo_clean --variant base \
          --ckpt-path "${repo_root}/policy/DP/checkpoints/${task}-demo_clean-200-${base_train_seed}/${checkpoint_num}.ckpt" \
          --rollout-dir "${rollout_dir}" --output-dir "${seed_eval_dir}" \
          --hard-seeds-file "${hard_seeds_file}" --policy-seed-offset 1000
      ) >"${log_dir}/eval_${task}_base_seed${seed}.log" 2>&1 &
      pids+=("$!")
      job=$((job + 1))
      if (( ${#pids[@]} == num_gpus )); then wait_jobs; fi

      for variant in "${variants[@]}"; do
        gpu=$((job % num_gpus))
        (
          cd "${repo_root}"
          export CUDA_VISIBLE_DEVICES="${gpu}"
          python "${phase_dir}/eval_per_seed.py" \
            --task "${task}" --task-config demo_clean --variant "${variant}" \
            --ckpt-path "${repo_root}/policy/DP/checkpoints/${task}-${variant}-${seed}/${epochs}.ckpt" \
            --rollout-dir "${rollout_dir}" --output-dir "${seed_eval_dir}" \
            --hard-seeds-file "${hard_seeds_file}" --policy-seed-offset 1000
        ) >"${log_dir}/eval_${task}_${variant}_seed${seed}.log" 2>&1 &
        pids+=("$!")
        job=$((job + 1))
        if (( ${#pids[@]} == num_gpus )); then wait_jobs; fi
      done
    done
    if (( ${#pids[@]} > 0 )); then wait_jobs; fi

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
