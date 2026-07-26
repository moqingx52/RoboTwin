#!/bin/bash
set -euo pipefail

# Saturate multi-GPU rollout collection without concurrent writes to a shard.
# Default: 8 GPUs x 3 workers/GPU = 24 processes, split evenly across 2 tasks.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

read -r -a tasks <<< "${PHASE3_TASKS:-place_container_plate dump_bin_bigbin}"
read -r -a gpu_ids <<< "${PHASE3_GPU_IDS:-0 1 2 3 4 5 6 7}"

workers_per_gpu=${PHASE3_ROLLOUT_WORKERS_PER_GPU:-3}
rollouts_per_seed=${PHASE3_ROLLOUTS_PER_SEED:-8}
expert_data_num=${PHASE3_EXPERT_DATA_NUM:-200}
checkpoint_num=${PHASE3_CHECKPOINT_NUM:-600}
train_seed=${PHASE3_BASE_TRAIN_SEED:-0}
action_dim=${PHASE3_ACTION_DIM:-14}
rollout_dir=${PHASE3_ROLLOUT_DIR:-experiments/phase1/rollouts_200}
log_dir=${PHASE3_ROLLOUT_LOG_DIR:-experiments/phase3/logs/failure_collection}

if (( workers_per_gpu < 1 )); then
  echo "PHASE3_ROLLOUT_WORKERS_PER_GPU must be >= 1" >&2
  exit 2
fi
if (( ${#gpu_ids[@]} == 0 || ${#tasks[@]} == 0 )); then
  echo "At least one GPU and task are required." >&2
  exit 2
fi

total_workers=$(( ${#gpu_ids[@]} * workers_per_gpu ))
if (( total_workers % ${#tasks[@]} != 0 )); then
  echo "GPU workers (${total_workers}) must divide evenly across tasks (${#tasks[@]})." >&2
  exit 2
fi
shards_per_task=$(( total_workers / ${#tasks[@]} ))

for task in "${tasks[@]}"; do
  seeds_file="experiments/phase1/seeds/${task}_seeds.json"
  checkpoint="policy/DP/checkpoints/${task}-demo_clean-${expert_data_num}-${train_seed}/${checkpoint_num}.ckpt"
  if [[ ! -s "${seeds_file}" ]]; then
    echo "Missing seeds file: ${seeds_file}" >&2
    exit 2
  fi
  if [[ ! -s "${checkpoint}" ]]; then
    echo "Missing base checkpoint: ${checkpoint}" >&2
    exit 2
  fi
done

# Never mix two layouts: merge_rollout_shards.py globs every shard count.
# Preserve incompatible manifests in a recoverable timestamped directory.
stamp="$(date +%Y%m%d_%H%M%S)"
for task in "${tasks[@]}"; do
  task_dir="${rollout_dir}/${task}"
  mkdir -p "${task_dir}"
  backup_dir="${task_dir}/layout_backup_${stamp}"
  moved=0

  shopt -s nullglob
  candidates=(
    "${task_dir}"/manifest.jsonl
    "${task_dir}"/seed_stats.json
    "${task_dir}"/manifest_shard_*_of_*.jsonl
    "${task_dir}"/seed_stats_shard_*_of_*.json
  )
  for path in "${candidates[@]}"; do
    [[ -e "${path}" ]] || continue
    name="$(basename "${path}")"
    if [[ "${name}" == *_of_$(printf '%02d' "${shards_per_task}").* ]]; then
      continue
    fi
    mkdir -p "${backup_dir}"
    mv "${path}" "${backup_dir}/"
    moved=1
  done
  shopt -u nullglob

  if (( moved )); then
    echo "Archived incompatible ${task} manifests to ${backup_dir}"
  fi
done

mkdir -p "${log_dir}"

pids=()
names=()
stopping=0

stop_children() {
  local signal=${1:-INT}
  if (( stopping )); then
    return
  fi
  stopping=1
  echo "Stopping ${#pids[@]} rollout workers with SIG${signal}..." >&2
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "-${signal}" "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}

trap 'stop_children INT; exit 130' INT TERM

job_index=0
for task in "${tasks[@]}"; do
  for ((shard=0; shard<shards_per_task; shard++)); do
    gpu="${gpu_ids[job_index % ${#gpu_ids[@]}]}"
    name="${task}_shard$(printf '%02d' "${shard}")_of_$(printf '%02d' "${shards_per_task}")"
    log_path="${log_dir}/${name}.log"

    CUDA_VISIBLE_DEVICES="${gpu}" \
    python experiments/phase1/collect_rollouts.py \
      --task "${task}" \
      --task-config demo_clean \
      --seeds-file "experiments/phase1/seeds/${task}_seeds.json" \
      --output-dir "${rollout_dir}" \
      --expert-data-num "${expert_data_num}" \
      --checkpoint-num "${checkpoint_num}" \
      --train-seed "${train_seed}" \
      --rollouts-per-seed "${rollouts_per_seed}" \
      --action-dim "${action_dim}" \
      --shard-id "${shard}" \
      --num-shards "${shards_per_task}" \
      --save-failures \
      --resume \
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

trap - INT TERM

if (( failed )); then
  echo "One or more rollout shards failed. Re-run this script to resume completed shard items." >&2
  exit 1
fi

for task in "${tasks[@]}"; do
  python experiments/phase1/verify_rollouts.py \
    --tasks "${task}" \
    --rollout-dir "${rollout_dir}" \
    --rollouts-per-seed "${rollouts_per_seed}" \
    --num-shards "${shards_per_task}" \
    --require-failures

  python experiments/phase1/merge_rollout_shards.py \
    --task "${task}" \
    --task-config demo_clean \
    --seeds-file "experiments/phase1/seeds/${task}_seeds.json" \
    --rollouts-per-seed "${rollouts_per_seed}" \
    --rollout-dir "${rollout_dir}"
done

echo "Failure rollout collection completed with ${shards_per_task} shards/task."
