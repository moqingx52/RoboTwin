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

log_root="${BRACE_LINE_B_LOG_DIR:-experiments/brace/logs/line_b_assets}"
mkdir -p "${log_root}"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
manifest_log="${log_root}/${run_stamp}_manifest.log"
echo "line_b_assets start ${run_stamp}" | tee "${manifest_log}"

patch_demo_clean_for_multitask() {
  if [[ ! -f task_config/demo_clean.yml.bak_line_b ]]; then
    cp task_config/demo_clean.yml task_config/demo_clean.yml.bak_line_b
  fi
  python - <<'PY'
from pathlib import Path
import yaml
path = Path("task_config/demo_clean.yml")
data = yaml.safe_load(path.read_text(encoding="utf-8"))
data["episode_num"] = 50
data["use_seed"] = True
path.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")
PY
}

restore_demo_clean() {
  if [[ -f task_config/demo_clean.yml.bak_line_b ]]; then
    mv task_config/demo_clean.yml.bak_line_b task_config/demo_clean.yml
  fi
}

trap restore_demo_clean EXIT
patch_demo_clean_for_multitask

collect_one() {
  local task=$1 gpu=$2
  local log="${log_root}/${run_stamp}_${task}.log"
  {
    echo "=== ${task} gpu=${gpu} ==="
    python experiments/brace/prepare_multitask_demo_seeds.py "${task}" --count 50
    local ckpt="policy/DP/checkpoints/${task}-demo_clean-50-0/600.ckpt"
    local zarr="policy/DP/data/${task}-demo_clean-50.zarr"
    if [[ -f "${ckpt}" && -d "${zarr}" ]]; then
      echo "skip collect/process/train: existing ${ckpt} and ${zarr}"
      return 0
    fi
    export CUDA_VISIBLE_DEVICES="${gpu}"
    bash collect_data.sh "${task}" demo_clean "${gpu}"
    (cd policy/DP && bash process_data.sh "${task}" demo_clean 50)
    (cd policy/DP && bash train.sh "${task}" demo_clean 50 0 14 "${gpu}")
    echo "=== ${task} complete ==="
  } >>"${log}" 2>&1
}

pids=()
names=()
for idx in "${!tasks[@]}"; do
  task="${tasks[idx]}"
  gpu="${gpu_ids[$((idx % ${#gpu_ids[@]}))]}"
  collect_one "${task}" "${gpu}" &
  pids+=("$!")
  names+=("${task}")
done

failed=0
for idx in "${!pids[@]}"; do
  if ! wait "${pids[idx]}"; then
    echo "FAILED ${names[idx]}" | tee -a "${manifest_log}"
    failed=$((failed + 1))
  else
    echo "OK ${names[idx]}" | tee -a "${manifest_log}"
  fi
done

echo "line_b_assets done failed=${failed}/${#tasks[@]}" | tee -a "${manifest_log}"
exit "${failed}"
