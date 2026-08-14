#!/bin/bash
# Scan rollout_train pools; per-task fail-closed, batch continues, summary at end.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

source /root/miniconda/etc/profile.d/conda.sh
conda activate RoboTwin
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

read -r -a scan_gpu_ids <<< "${BRACE_GPU_IDS:-0 1 2 3 4 5 6 7}"
if [[ -z "${BRACE_FEASIBILITY_TASKS+x}" ]]; then
  read -r -a tasks <<< "beat_block_hammer handover_mic lift_pot open_laptop place_burger_fries shake_bottle stack_bowls_three"
else
  read -r -a tasks <<< "${BRACE_FEASIBILITY_TASKS}"
fi
read -r -a summary_tasks <<< "${BRACE_FEASIBILITY_SUMMARY_TASKS:-${tasks[*]}}"
verify_task="${BRACE_FEASIBILITY_VERIFY_TASK:-}"
verify_gpu="${BRACE_FEASIBILITY_VERIFY_GPU:-}"
num_shards=${BRACE_FEASIBILITY_SHARDS:-1}
# Deterministic re-verify is a diagnostic isolation run: one probe process per
# GPU so the observed outcome cannot be caused by concurrent simulator workers.
verify_shards=${BRACE_FEASIBILITY_VERIFY_SHARDS:-1}
run_dir="${BRACE_FEASIBILITY_RUN_DIR:-experiments/brace/runs/seed_feasibility_$(date -u +%Y%m%dT%H%M%SZ)}"
force_rescan="${BRACE_FEASIBILITY_FORCE:-0}"
probe_repeats=${BRACE_FEASIBILITY_PROBE_REPEATS:-1}
probe_success_rule=${BRACE_FEASIBILITY_PROBE_SUCCESS_RULE:-single}
mkdir -p "${run_dir}" experiments/brace/logs/seed_feasibility

log_file="${BRACE_FEASIBILITY_LOG:-experiments/brace/logs/seed_feasibility/run_$(basename "${run_dir}").log}"
exec > >(tee -a "${log_file}") 2>&1

read -r -a supplement_tasks <<< "${BRACE_FEASIBILITY_SUPPLEMENT_TASKS:-}"
supplement_shards=${BRACE_FEASIBILITY_SUPPLEMENT_SHARDS:-3}
echo "seed_feasibility run_dir=${run_dir} tasks=${tasks[*]} shards=${num_shards} verify_task=${verify_task:-none} verify_shards=${verify_shards} supplement_tasks=${supplement_tasks[*]:-none} supplement_shards=${supplement_shards}"

# Record scheduler-state metadata: actual GPU IDs, workers and workers-per-GPU.
python experiments/brace/write_run_meta.py \
  --run-dir "${run_dir}" \
  --stage "seed_feasibility_scan" \
  --run-id "$(basename "${run_dir}")" \
  --tasks "${tasks[@]}" \
  --extra "gpu_ids=${scan_gpu_ids[*]}" "workers_per_gpu=3" "scan_shards=${num_shards}" \
          "verify_task=${verify_task:-none}" "verify_gpu=${verify_gpu:-none}" \
          "verify_shards=${verify_shards}" "verify_workers_per_gpu=1" \
          "supplement_tasks=${supplement_tasks[*]:-none}" "supplement_shards=${supplement_shards}" \
          "force_rescan=${force_rescan}" "probe_repeats=${probe_repeats}" \
          "probe_success_rule=${probe_success_rule}"

scan_one_task() {
  local task=$1 gpu=$2 verify_label=${3:-}
  local shards=${4:-${num_shards}}
  local candidate_partition=${5:-}
  local shard_dir="${run_dir}/${task}"
  if [[ -n "${verify_label}" ]]; then
    shard_dir="${run_dir}/${task}_${verify_label}"
  elif [[ -n "${candidate_partition}" ]]; then
    shard_dir="${run_dir}/${task}_supplement"
  fi
  local output="${run_dir}/${task}_feasibility.json"
  if [[ -n "${verify_label}" ]]; then
    output="${run_dir}/${task}_${verify_label}_feasibility.json"
  elif [[ -n "${candidate_partition}" ]]; then
    output="${run_dir}/${task}_supplement_shards.json"
  fi

  if [[ "${force_rescan}" == "1" ]]; then
    # Remove the exact target so stale shards cannot be globbed into the merge.
    rm -rf "${shard_dir}"
    rm -f "${output}"
  elif [[ -f "${output}" ]]; then
    echo "SKIP ${task} verify_label=${verify_label:-none} partition=${candidate_partition:-default}: existing ${output}"
    return 0
  fi

  mkdir -p "${shard_dir}"

  if (( shards == 1 )); then
    local single_output="${output}"
    local -a single_args=(
      --task "${task}"
      --shard-id 0
      --num-shards 1
      --gpu-id "${gpu}"
      --probe-repeats "${probe_repeats}"
      --probe-success-rule "${probe_success_rule}"
    )
    if [[ -n "${candidate_partition}" ]]; then
      # Supplement scans must land as a shard payload so the combine step can
      # consume them uniformly, even with a single shard.
      single_args+=(--write-shard)
      single_output="${shard_dir}/shard_00_of_01.json"
    fi
    single_args+=(--output "${single_output}")
    if [[ -n "${verify_label}" ]]; then
      single_args+=(--verify-label "${verify_label}")
    fi
    if [[ -n "${candidate_partition}" ]]; then
      single_args+=(--candidate-partition "${candidate_partition}")
    elif [[ "${BRACE_FEASIBILITY_PROVISIONAL_MANIFEST:-1}" == "1" ]]; then
      single_args+=(--provisional-manifest-update)
    fi
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      python experiments/brace/scan_seed_feasibility.py "${single_args[@]}"
    )
    local single_status=$?
    if (( single_status != 0 )); then
      echo "TASK_SCAN_FAILURE ${task} verify_label=${verify_label:-none} partition=${candidate_partition:-default}" >&2
      return 1
    fi
    echo "TASK_DONE ${task} verify_label=${verify_label:-none} partition=${candidate_partition:-default} output=${single_output}"
    return 0
  fi

  local -a shard_pids=()
  local shard=0
  for ((shard=0; shard<shards; shard++)); do
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      args=(
        --task "${task}"
        --shard-id "${shard}"
        --num-shards "${shards}"
        --gpu-id "${gpu}"
        --probe-repeats "${probe_repeats}"
        --probe-success-rule "${probe_success_rule}"
        --output "${shard_dir}/shard_$(printf '%02d' "${shard}")_of_$(printf '%02d' "${shards}").json"
      )
      if [[ -n "${verify_label}" ]]; then
        args+=(--verify-label "${verify_label}")
      fi
      if [[ -n "${candidate_partition}" ]]; then
        args+=(--candidate-partition "${candidate_partition}")
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
    echo "TASK_SHARD_FAILURE ${task} verify_label=${verify_label:-none} partition=${candidate_partition:-default} failed_shards=${shard_failed}" >&2
    return 1
  fi

  local -a merge_inputs=()
  local shard_path
  for shard_path in "${shard_dir}"/shard_*_of_*.json; do
    merge_inputs+=("${shard_path}")
  done
  if (( ${#merge_inputs[@]} != shards )); then
    echo "TASK_SHARD_COUNT ${task} verify_label=${verify_label:-none}: expected ${shards} shard files, found ${#merge_inputs[@]}" >&2
    return 1
  fi

  local -a merge_args=(
    --task "${task}"
    --merge-inputs "${merge_inputs[@]}"
    --output "${output}"
    --gpu-id "${gpu}"
    --probe-repeats "${probe_repeats}"
    --probe-success-rule "${probe_success_rule}"
  )
  if [[ -n "${verify_label}" ]]; then
    merge_args+=(--verify-label "${verify_label}")
  elif [[ -n "${candidate_partition}" ]]; then
    merge_args+=(--candidate-partition "${candidate_partition}")
  elif [[ "${BRACE_FEASIBILITY_PROVISIONAL_MANIFEST:-1}" == "1" ]]; then
    merge_args+=(--provisional-manifest-update)
  fi

  if ! python experiments/brace/scan_seed_feasibility.py "${merge_args[@]}"; then
    echo "TASK_MERGE_FAILURE ${task} verify_label=${verify_label:-none} partition=${candidate_partition:-default}" >&2
    return 1
  fi
  echo "TASK_DONE ${task} verify_label=${verify_label:-none} partition=${candidate_partition:-default} output=${output}"
  return 0
}

# Build the queue of tasks that still need evidence.
declare -a pending_tasks=()
for task in "${tasks[@]}"; do
  evidence="${run_dir}/${task}_feasibility.json"
  if [[ "${force_rescan}" != "1" && -f "${evidence}" ]]; then
    echo "SKIP queued ${task}: existing evidence"
    continue
  fi
  pending_tasks+=("${task}")
done

# Schedule in GPU-sized waves: never break on overflow, later tasks go to the
# next wave. One scan job per GPU; each job fans out to 3 shards on that GPU.
task_failed=0
launch_wave() {
  local -a wave_tasks=("${@}")
  local -a pids=()
  local -a names=()
  local -a assigned_gpus=()
  local task gpu idx
  for task in "${wave_tasks[@]}"; do
    gpu=""
    for candidate in "${scan_gpu_ids[@]}"; do
      local taken=0
      for used in "${assigned_gpus[@]}"; do
        if [[ "${used}" == "${candidate}" ]]; then
          taken=1
        fi
      done
      if (( taken == 0 )); then
        gpu="${candidate}"
        break
      fi
    done
    if [[ -z "${gpu}" ]]; then
      echo "no free GPU for ${task}" >&2
      task_failed=$((task_failed + 1))
      continue
    fi
    assigned_gpus+=("${gpu}")
    echo "LAUNCH ${task} gpu=${gpu}"
    scan_one_task "${task}" "${gpu}" &
    pids+=("$!")
    names+=("${task}")
  done
  for idx in "${!pids[@]}"; do
    if ! wait "${pids[idx]}"; then
      echo "FAILED ${names[idx]}"
      task_failed=$((task_failed + 1))
    else
      echo "OK ${names[idx]}"
    fi
  done
}

for ((offset = 0; offset < ${#pending_tasks[@]}; offset += ${#scan_gpu_ids[@]})); do
  wave=("${pending_tasks[@]:offset:${#scan_gpu_ids[@]}}")
  launch_wave "${wave[@]}"
done

# Supplement scan (brace.multitask.v1.1): only for tasks whose original pool
# was insufficient. Original evidence is bound by SHA; the combined evidence is
# written to ${task}_supplemented_feasibility.json and the original stays intact.
combine_supplement() {
  local task=$1 gpu=$2
  local primary="${run_dir}/${task}_feasibility.json"
  if [[ ! -f "${primary}" ]]; then
    echo "SUPPLEMENT_SKIP ${task}: missing primary evidence ${primary}" >&2
    task_failed=$((task_failed + 1))
    return 1
  fi
  echo "SUPPLEMENT_LAUNCH ${task} gpu=${gpu} shards=${supplement_shards} partition=expert_demo_supplement"
  if ! scan_one_task "${task}" "${gpu}" "" "${supplement_shards}" "expert_demo_supplement"; then
    echo "SUPPLEMENT_FAILED ${task}" >&2
    task_failed=$((task_failed + 1))
    return 1
  fi
  local -a shard_inputs=()
  local shard_path
  for shard_path in "${run_dir}/${task}_supplement"/shard_*_of_*.json; do
    shard_inputs+=("${shard_path}")
  done
  local -a combine_args=(
    --task "${task}"
    --original-evidence "${primary}"
    --merge-inputs "${shard_inputs[@]}"
    --output "${run_dir}/${task}_supplemented_feasibility.json"
    --gpu-id "${gpu}"
    --probe-repeats "${probe_repeats}"
    --probe-success-rule "${probe_success_rule}"
  )
  if [[ "${BRACE_FEASIBILITY_PROVISIONAL_MANIFEST:-1}" == "1" ]]; then
    combine_args+=(--provisional-manifest-update)
  fi
  if ! python experiments/brace/scan_seed_feasibility.py "${combine_args[@]}"; then
    echo "SUPPLEMENT_COMBINE_FAILURE ${task}" >&2
    task_failed=$((task_failed + 1))
    return 1
  fi
  echo "SUPPLEMENT_DONE ${task} output=${run_dir}/${task}_supplemented_feasibility.json"
  return 0
}

if ((${#supplement_tasks[@]} > 0)); then
  echo "supplement pool scan for: ${supplement_tasks[*]}"
  for task in "${supplement_tasks[@]}"; do
    combine_supplement "${task}" "${scan_gpu_ids[0]}"
  done
fi

# Optional deterministic re-verify (e.g. lift_pot): one shard per GPU, so the
# outcome reflects a single probe process (diagnostic isolation run).
if [[ -n "${verify_task}" && -n "${verify_gpu}" ]]; then
  echo "LAUNCH verify ${verify_task} gpu=${verify_gpu} shards=${verify_shards} (diagnostic isolation, 1 shard/GPU)"
  scan_one_task "${verify_task}" "${verify_gpu}" "reverify" "${verify_shards}" &
  verify_pid=$!
  if ! wait "${verify_pid}"; then
    echo "FAILED ${verify_task}:reverify"
    task_failed=$((task_failed + 1))
  else
    echo "OK ${verify_task}:reverify"
  fi
fi

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
