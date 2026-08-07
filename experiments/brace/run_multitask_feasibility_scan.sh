#!/bin/bash
# Scan rollout_train pools; per-task fail-closed, batch continues, summary at end.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

source /root/miniconda/etc/profile.d/conda.sh
conda activate RoboTwin
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

read -r -a scan_gpu_ids <<< "${BRACE_GPU_IDS:-0 1 2 3 4 5 6 7}"
read -r -a tasks <<< "${BRACE_FEASIBILITY_TASKS:-beat_block_hammer handover_mic lift_pot open_laptop place_burger_fries shake_bottle stack_bowls_three}"
read -r -a summary_tasks <<< "${BRACE_FEASIBILITY_SUMMARY_TASKS:-${tasks[*]}}"
verify_task="${BRACE_FEASIBILITY_VERIFY_TASK:-}"
verify_gpu="${BRACE_FEASIBILITY_VERIFY_GPU:-}"
num_shards=${BRACE_FEASIBILITY_SHARDS:-1}
verify_shards=${BRACE_FEASIBILITY_VERIFY_SHARDS:-4}
run_dir="${BRACE_FEASIBILITY_RUN_DIR:-experiments/brace/runs/seed_feasibility_$(date -u +%Y%m%dT%H%M%SZ)}"
force_rescan="${BRACE_FEASIBILITY_FORCE:-0}"
mkdir -p "${run_dir}" experiments/brace/logs/seed_feasibility

log_file="${BRACE_FEASIBILITY_LOG:-experiments/brace/logs/seed_feasibility/run_$(basename "${run_dir}").log}"
exec > >(tee -a "${log_file}") 2>&1

echo "seed_feasibility run_dir=${run_dir} tasks=${tasks[*]} shards=${num_shards} verify_task=${verify_task:-none}"

scan_one_task() {
  local task=$1 gpu=$2 verify_label=${3:-}
  local shards=${4:-${num_shards}}
  local shard_dir="${run_dir}/${task}"
  if [[ -n "${verify_label}" ]]; then
    shard_dir="${run_dir}/${task}_${verify_label}"
  fi
  local output="${run_dir}/${task}_feasibility.json"
  if [[ -n "${verify_label}" ]]; then
    output="${run_dir}/${task}_${verify_label}_feasibility.json"
  fi

  if [[ "${force_rescan}" != "1" && -f "${output}" ]]; then
    echo "SKIP ${task} verify_label=${verify_label:-none}: existing ${output}"
    return 0
  fi

  mkdir -p "${shard_dir}"
  local -a shard_pids=()
  local shard=0
  for ((shard=0; shard<shards; shard++)); do
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      args=(
        --task "${task}"
        --shard-id "${shard}"
        --num-shards "${shards}"
        --output "${shard_dir}/shard_$(printf '%02d' "${shard}")_of_$(printf '%02d' "${shards}").json"
      )
      if [[ -n "${verify_label}" ]]; then
        args+=(--verify-label "${verify_label}")
      fi
      python experiments/brace/scan_seed_feasibility.py "${args[@]}"
    ) &
    shard_pids+=("$!")
  done

  local shard_failed=0
  for pid in "${shard_pids[@]}"; do
    if ! wait "${pid}"; then
      shard_failed=$((shard_failed + 1))
    fi
  done
  if (( shard_failed > 0 )); then
    echo "TASK_SHARD_FAILURE ${task} verify_label=${verify_label:-none} failed_shards=${shard_failed}" >&2
    return 1
  fi

  local -a merge_inputs=()
  local shard_path
  for shard_path in "${shard_dir}"/shard_*_of_*.json; do
    merge_inputs+=("${shard_path}")
  done

  local -a merge_args=(
    --task "${task}"
    --merge-inputs "${merge_inputs[@]}"
    --output "${output}"
  )
  if [[ -n "${verify_label}" ]]; then
    merge_args+=(--verify-label "${verify_label}")
  elif [[ "${BRACE_FEASIBILITY_PROVISIONAL_MANIFEST:-1}" == "1" ]]; then
    merge_args+=(--provisional-manifest-update)
  fi

  if ! python experiments/brace/scan_seed_feasibility.py "${merge_args[@]}"; then
    echo "TASK_MERGE_FAILURE ${task} verify_label=${verify_label:-none}" >&2
    return 1
  fi
  echo "TASK_DONE ${task} verify_label=${verify_label:-none} output=${output}"
  return 0
}

# Remaining scan tasks: assign one primary GPU each (up to four in parallel).
declare -a active_pids=()
declare -a active_names=()
task_index=0
for task in "${tasks[@]}"; do
  evidence="${run_dir}/${task}_feasibility.json"
  if [[ "${force_rescan}" != "1" && -f "${evidence}" ]]; then
    echo "SKIP queued ${task}: existing evidence"
    continue
  fi
  if (( task_index >= ${#scan_gpu_ids[@]} )); then
    break
  fi
  gpu="${scan_gpu_ids[task_index]}"
  echo "LAUNCH ${task} gpu=${gpu}"
  scan_one_task "${task}" "${gpu}" &
  active_pids+=("$!")
  active_names+=("${task}")
  task_index=$((task_index + 1))
done

# Optional deterministic re-verify (e.g. lift_pot on GPU 4).
if [[ -n "${verify_task}" && -n "${verify_gpu}" ]]; then
  echo "LAUNCH verify ${verify_task} gpu=${verify_gpu} shards=${verify_shards}"
  scan_one_task "${verify_task}" "${verify_gpu}" "reverify" "${verify_shards}" &
  active_pids+=("$!")
  active_names+=("${verify_task}:reverify")
fi

task_failed=0
for idx in "${!active_pids[@]}"; do
  if ! wait "${active_pids[idx]}"; then
    echo "FAILED ${active_names[idx]}"
    task_failed=$((task_failed + 1))
  else
    echo "OK ${active_names[idx]}"
  fi
done

if ! python experiments/brace/aggregate_seed_feasibility_batch.py \
  --run-dir "${run_dir}" \
  --tasks "${summary_tasks[@]}" \
  --output "${run_dir}/batch_summary.json"; then
  echo "batch_summary: unresolved tasks remain"
  exit 1
fi

if (( task_failed > 0 )); then
  echo "seed_feasibility finished with task launch/merge failures=${task_failed}"
  exit 1
fi

echo "seed_feasibility complete run_dir=${run_dir}"
