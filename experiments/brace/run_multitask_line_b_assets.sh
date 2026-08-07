#!/bin/bash
# Line B: build held-out multitask base assets (50-demo data → zarr → DP 600.ckpt).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

source /root/miniconda/etc/profile.d/conda.sh
conda activate RoboTwin
export PYTHONPATH="${repo_root}:${repo_root}/policy/DP${PYTHONPATH:+:${PYTHONPATH}}"

read -r -a gpu_ids <<< "${BRACE_GPU_IDS:-0 1 2 3 4 5 6 7}"
read -r -a tasks <<< "${BRACE_HELDOUT_TASKS:-beat_block_hammer click_alarmclock handover_mic lift_pot move_can_pot open_laptop place_burger_fries put_object_cabinet shake_bottle stack_bowls_three}"
required_episodes=${BRACE_EXPERT_DEMO_COUNT:-50}

if ((${#gpu_ids[@]} == 0)); then
  echo "BRACE_GPU_IDS must contain at least one GPU" >&2
  exit 2
fi
declare -A seen_gpu_ids=()
for gpu in "${gpu_ids[@]}"; do
  if [[ -n "${seen_gpu_ids[${gpu}]+present}" ]]; then
    echo "BRACE_GPU_IDS contains duplicate GPU ${gpu}" >&2
    exit 2
  fi
  seen_gpu_ids["${gpu}"]=1
done

log_root="${BRACE_LINE_B_LOG_DIR:-experiments/brace/logs/line_b_assets}"
mkdir -p "${log_root}"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
manifest_log="${log_root}/${run_stamp}_manifest.log"
echo "line_b_assets start ${run_stamp}" | tee "${manifest_log}"

patch_demo_clean_for_multitask() {
  if [[ ! -f task_config/demo_clean.yml.bak_line_b ]]; then
    cp task_config/demo_clean.yml task_config/demo_clean.yml.bak_line_b
  fi
  python - <<PY
from pathlib import Path
import yaml
path = Path("task_config/demo_clean.yml")
data = yaml.safe_load(path.read_text(encoding="utf-8"))
data["episode_num"] = ${required_episodes}
data["use_seed"] = True
path.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")
PY
}

restore_demo_clean() {
  if [[ -f task_config/demo_clean.yml.bak_line_b ]]; then
    mv task_config/demo_clean.yml.bak_line_b task_config/demo_clean.yml
  fi
}

count_demo_hdf5() {
  local task=$1
  local data_dir="data/${task}/demo_clean/data"
  if [[ ! -d "${data_dir}" ]]; then
    echo 0
    return 0
  fi
  find "${data_dir}" -maxdepth 1 -type f -name 'episode*.hdf5' | wc -l | tr -d ' '
}

remove_empty_demo_zarr() {
  local task=$1
  local zarr="policy/DP/data/${task}-demo_clean-${required_episodes}.zarr"
  if [[ -d "${zarr}" ]] && ! python - <<PY
import sys
import zarr
root = zarr.open("${zarr}", mode="r")
sys.exit(0 if "episode_ends" in root["meta"] else 1)
PY
  then
    rm -rf "${zarr}"
    echo "removed invalid zarr ${zarr}"
  fi
}

trap restore_demo_clean EXIT
patch_demo_clean_for_multitask

collect_one() {
  local task=$1 gpu=$2
  local log="${log_root}/${run_stamp}_${task}.log"
  {
    set -euo pipefail
    echo "=== ${task} gpu=${gpu} ==="
    local ckpt="policy/DP/checkpoints/${task}-demo_clean-${required_episodes}-0/600.ckpt"
    local zarr="policy/DP/data/${task}-demo_clean-${required_episodes}.zarr"
    if [[ -f "${ckpt}" && -d "${zarr}" ]]; then
      echo "skip collect/process/train: existing ${ckpt} and ${zarr}"
      exit 0
    fi
    remove_empty_demo_zarr "${task}"
    python experiments/brace/prepare_multitask_demo_seeds.py "${task}" --count "${required_episodes}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    if ! bash collect_data.sh "${task}" demo_clean "${gpu}"; then
      echo "collect failed for ${task}" >&2
      exit 1
    fi
    local hdf5_count
    hdf5_count="$(count_demo_hdf5 "${task}")"
    if (( hdf5_count < required_episodes )); then
      echo "expected ${required_episodes} demo HDF5 for ${task}, found ${hdf5_count}" >&2
      exit 1
    fi
    if ! (cd policy/DP && bash process_data.sh "${task}" demo_clean "${required_episodes}"); then
      echo "process_data failed for ${task}" >&2
      exit 1
    fi
    if ! (cd policy/DP && bash train.sh "${task}" demo_clean "${required_episodes}" 0 14 "${gpu}"); then
      echo "train failed for ${task}" >&2
      exit 1
    fi
    echo "=== ${task} complete ==="
  } >>"${log}" 2>&1
}

failed=0
completed=0
pending_tasks=()
for task in "${tasks[@]}"; do
  ckpt="policy/DP/checkpoints/${task}-demo_clean-${required_episodes}-0/600.ckpt"
  zarr="policy/DP/data/${task}-demo_clean-${required_episodes}.zarr"
  if [[ -f "${ckpt}" && -d "${zarr}" ]]; then
    echo "SKIP ${task}: checkpoint and zarr already exist" | tee -a "${manifest_log}"
    completed=$((completed + 1))
  else
    pending_tasks+=("${task}")
  fi
done

echo "schedule pending=${#pending_tasks[@]} gpus=${gpu_ids[*]} one_task_per_gpu=true" | tee -a "${manifest_log}"

for ((offset = 0; offset < ${#pending_tasks[@]}; offset += ${#gpu_ids[@]})); do
  pids=()
  names=()
  for gpu_idx in "${!gpu_ids[@]}"; do
    task_idx=$((offset + gpu_idx))
    if ((task_idx >= ${#pending_tasks[@]})); then
      break
    fi
    task="${pending_tasks[task_idx]}"
    gpu="${gpu_ids[gpu_idx]}"
    echo "LAUNCH ${task} gpu=${gpu}" | tee -a "${manifest_log}"
    collect_one "${task}" "${gpu}" &
    pids+=("$!")
    names+=("${task}")
  done

  for idx in "${!pids[@]}"; do
    if ! wait "${pids[idx]}"; then
      echo "FAILED ${names[idx]}" | tee -a "${manifest_log}"
      failed=$((failed + 1))
    else
      echo "OK ${names[idx]}" | tee -a "${manifest_log}"
      completed=$((completed + 1))
    fi
  done
done

echo "line_b_assets done completed=${completed}/${#tasks[@]} failed=${failed}/${#tasks[@]}" | tee -a "${manifest_log}"
if ((failed > 0)); then
  exit 1
fi
