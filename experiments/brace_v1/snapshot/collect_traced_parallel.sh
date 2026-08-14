#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

read -r -a tasks <<< "${BRACE_TASKS:-place_container_plate dump_bin_bigbin}"
read -r -a gpu_ids <<< "${BRACE_GPU_IDS:-0 1 2 3 4 5 6 7}"
brace_dir=experiments/brace
# shellcheck source=experiments/brace/run_paths.sh
source "${repo_root}/experiments/brace/run_paths.sh"

workers_per_gpu=${BRACE_ROLLOUT_WORKERS_PER_GPU:-3}
rollouts_per_seed=${BRACE_ROLLOUTS_PER_SEED:-8}
expert_data_num=${BRACE_EXPERT_DATA_NUM:-200}
checkpoint_num=${BRACE_CHECKPOINT_NUM:-600}
train_seed=${BRACE_BASE_TRAIN_SEED:-0}
action_dim=${BRACE_ACTION_DIM:-14}
rollout_dir=${BRACE_TRACED_ROLLOUT_DIR:-experiments/brace/rollouts_traced}
log_dir=${BRACE_TRACED_LOG_DIR:-experiments/brace/logs/traced_collection}
task_config=${BRACE_TASK_CONFIG:-demo_brace_trace}
max_trajectories=${BRACE_TRACE_MAX_TRAJECTORIES:-}
env_seeds=${BRACE_TRACE_ENV_SEEDS:-}
rollout_ids=${BRACE_TRACE_ROLLOUT_IDS:-}
snapshots_per_trajectory=${BRACE_SNAPSHOTS_PER_TRAJECTORY:-3}

total_workers=$(( ${#gpu_ids[@]} * workers_per_gpu ))
shards_per_task=$(( total_workers / ${#tasks[@]} ))

resolve_pilot_shards() {
  local seeds_file=$1
  local seed_count
  seed_count="$(jq '.seeds | length' "${seeds_file}")"
  if [[ -n "${BRACE_PILOT_NUM_SHARDS:-}" ]]; then
    shards_per_task="${BRACE_PILOT_NUM_SHARDS}"
  elif (( seed_count > 0 )); then
    max_parallel=$(( ${#gpu_ids[@]} * workers_per_gpu ))
    if (( max_parallel > seed_count )); then
      shards_per_task="${seed_count}"
    elif (( max_parallel < seed_count )); then
      # Prefer one shard per GPU slot when seeds exceed parallel capacity.
      shards_per_task="${max_parallel}"
    fi
  fi
  if (( shards_per_task < 1 )); then
    shards_per_task=1
  fi
  total_workers=$(( ${#gpu_ids[@]} * workers_per_gpu ))
  if (( total_workers > shards_per_task )); then
    total_workers="${shards_per_task}"
  fi
}

mkdir -p "${log_dir}"
pids=()
names=()

job_index=0
for task in "${tasks[@]}"; do
  seeds_file="experiments/phase1/seeds/${task}_seeds.json"
  if [[ "${rollout_dir}" == *rollouts_traced_pilot* ]]; then
    if [[ -n "${BRACE_PILOT_SEEDS_FILE:-}" ]]; then
      seeds_file="${BRACE_PILOT_SEEDS_FILE}"
    elif ! seeds_file="$(brace_resolve_seeds_file "${task}_pilot_seeds.json")"; then
      seeds_file="experiments/brace/seeds/${task}_pilot_seeds.json"
    fi
    if [[ ! -s "${seeds_file}" ]]; then
      echo "Missing pilot seeds file: ${seeds_file}. Run select-pilot-seeds first." >&2
      exit 2
    fi
    resolve_pilot_shards "${seeds_file}"
    env_seeds=""
  fi
  for ((shard=0; shard<shards_per_task; shard++)); do
    gpu="${gpu_ids[job_index % ${#gpu_ids[@]}]}"
    name="${task}_shard$(printf '%02d' "${shard}")_of_$(printf '%02d' "${shards_per_task}")"
    log_path="${log_dir}/${name}.log"
    extra_args=()
    if [[ -n "${max_trajectories}" ]]; then
      extra_args+=(--max-trajectories "${max_trajectories}")
    fi
    if [[ -n "${env_seeds}" ]]; then
      extra_args+=(--env-seeds ${env_seeds})
    fi
    if [[ -n "${rollout_ids}" ]]; then
      extra_args+=(--rollout-ids ${rollout_ids})
    fi

    CUDA_VISIBLE_DEVICES="${gpu}" \
    python experiments/brace/collect_traced_rollouts.py \
      --task "${task}" \
      --task-config "${task_config}" \
      --seeds-file "${seeds_file}" \
      --output-dir "${rollout_dir}" \
      --expert-data-num "${expert_data_num}" \
      --checkpoint-num "${checkpoint_num}" \
      --train-seed "${train_seed}" \
      --rollouts-per-seed "${rollouts_per_seed}" \
      --action-dim "${action_dim}" \
      --shard-id "${shard}" \
      --num-shards "${shards_per_task}" \
      --snapshots-per-trajectory "${snapshots_per_trajectory}" \
      --save-failures \
      --resume \
      "${extra_args[@]}" \
      >"${log_path}" 2>&1 &

    pids+=("$!")
    names+=("${name}")
    echo "GPU ${gpu}: started ${name}, pid=$!"
    job_index=$((job_index + 1))
  done
done

failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[i]}"; then
    echo "OK: ${names[i]}"
  else
    echo "FAILED: ${names[i]} (see ${log_dir}/${names[i]}.log)" >&2
    failed=1
  fi
done

if (( failed )); then
  exit 1
fi

if [[ -n "${max_trajectories}" ]]; then
  echo "Smoke/partial collection finished (${max_trajectories} trajectories cap); skipping full verify."
  exit 0
fi

for task in "${tasks[@]}"; do
  verify_seeds_file="experiments/phase1/seeds/${task}_seeds.json"
  if [[ "${rollout_dir}" == *rollouts_traced_pilot* ]]; then
    if [[ -n "${BRACE_PILOT_SEEDS_FILE:-}" ]]; then
      verify_seeds_file="${BRACE_PILOT_SEEDS_FILE}"
    elif ! verify_seeds_file="$(brace_resolve_seeds_file "${task}_pilot_seeds.json")"; then
      verify_seeds_file="experiments/brace/seeds/${task}_pilot_seeds.json"
    fi
  fi
  python experiments/brace/verify_traced_rollouts.py \
    --tasks "${task}" \
    --rollout-dir "${rollout_dir}" \
    --seeds-file "${verify_seeds_file}" \
    --rollouts-per-seed "${rollouts_per_seed}" \
    --num-shards "${shards_per_task}" \
    --require-failures \
    --workers "${BRACE_VERIFY_WORKERS:-96}"

  python experiments/phase1/merge_rollout_shards.py \
    --task "${task}" \
    --task-config "${task_config}" \
    --seeds-file "${verify_seeds_file}" \
    --rollouts-per-seed "${rollouts_per_seed}" \
    --rollout-dir "${rollout_dir}"
done

echo "Traced rollout collection completed with ${shards_per_task} shards/task."
