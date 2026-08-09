#!/bin/bash
# Line B: build held-out multitask base assets (50-demo data → zarr → DP 600.ckpt).
#
# Scheduling:
#   - Default task list excludes put_object_cabinet (protocol-feasibility failure).
#   - Stages: collect | process | train | all (default all runs sequentially).
#   - collect: up to BRACE_LINE_B_COLLECT_WORKERS_PER_GPU parallel simulators/GPU (default 3).
#   - train: one DP job per GPU (exclusive; do not colocate with collection).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

source /root/miniconda/etc/profile.d/conda.sh
conda activate RoboTwin
export PYTHONPATH="${repo_root}:${repo_root}/policy/DP${PYTHONPATH:+:${PYTHONPATH}}"

read -r -a gpu_ids <<< "${BRACE_GPU_IDS:-0 1 2 3 4 5 6 7}"
# Nine frozen held-out tasks; put_object_cabinet blocked pending protocol decision.
read -r -a tasks <<< "${BRACE_HELDOUT_TASKS:-beat_block_hammer click_alarmclock handover_mic lift_pot move_can_pot open_laptop place_burger_fries shake_bottle stack_bowls_three}"
required_episodes=${BRACE_EXPERT_DEMO_COUNT:-50}
line_b_stage=${BRACE_LINE_B_STAGE:-all}
collect_workers_per_gpu=${BRACE_LINE_B_COLLECT_WORKERS_PER_GPU:-3}
train_workers_per_gpu=${BRACE_LINE_B_TRAIN_WORKERS_PER_GPU:-1}
premotion_max_attempts=${BRACE_PREMOTION_MAX_ATTEMPTS:-1}
premotion_retry_amendment=${BRACE_PREMOTION_RETRY_AMENDMENT:-}
export BRACE_PREMOTION_MAX_ATTEMPTS="${premotion_max_attempts}"
if [[ -n "${premotion_retry_amendment}" ]]; then
  export BRACE_PREMOTION_RETRY_AMENDMENT="${premotion_retry_amendment}"
fi
premotion_retry_amendment_sha256=""
if [[ "${premotion_max_attempts}" != "1" ]]; then
  for task in "${tasks[@]}"; do
    python - "${task}" <<'PY'
import sys
from pathlib import Path
from experiments.brace.premotion_retry import resolve_max_attempts

attempts, amendment = resolve_max_attempts(cwd=Path.cwd(), task_name=sys.argv[1])
print(f"validated bounded pre-motion retry task={sys.argv[1]} attempts={attempts} amendment={amendment}")
PY
  done
fi
if [[ -n "${premotion_retry_amendment}" && -f "${premotion_retry_amendment}" ]]; then
  premotion_retry_amendment_sha256="$(python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "${premotion_retry_amendment}")"
fi
run_dir="${BRACE_LINE_B_RUN_DIR:-experiments/brace/runs/line_b_assets_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${run_dir}"

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
run_stamp="$(basename "${run_dir}")"
manifest_log="${log_root}/${run_stamp}_manifest.log"
echo "line_b_assets start ${run_stamp} stage=${line_b_stage}" | tee "${manifest_log}"

python experiments/brace/write_run_meta.py \
  --run-dir "${run_dir}" \
  --stage "line_b_assets_${line_b_stage}" \
  --run-id "${run_stamp}" \
  --tasks "${tasks[@]}" \
  --extra \
    "gpu_ids=${gpu_ids[*]}" \
    "collect_workers_per_gpu=${collect_workers_per_gpu}" \
    "train_workers_per_gpu=${train_workers_per_gpu}" \
    "premotion_max_attempts=${premotion_max_attempts}" \
    "premotion_retry_amendment=${premotion_retry_amendment}" \
    "premotion_retry_amendment_sha256=${premotion_retry_amendment_sha256}" \
    "line_b_stage=${line_b_stage}" \
    "scheduling_note=collect_uses_${collect_workers_per_gpu}_sim_workers_per_gpu;train_exclusive_${train_workers_per_gpu}_per_gpu" \
    "partial_line_b=true" \
    "formal_preflight_requires_10_of_10=false_until_put_object_cabinet_resolved"

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

manifest_sha256() {
  python -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$1"
}

expected_cohort_seeds() {
  local task=$1
  python - "$1" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads(Path(f"experiments/brace/seeds/multitask_v1/{sys.argv[1]}.json").read_text(encoding="utf-8"))
seeds = manifest.get("cohorts", {}).get("expert_demo", [])
print(" ".join(str(int(s)) for s in seeds))
PY
}

cohort_matches_manifest() {
  local task=$1
  local prov="data/${task}/demo_clean/cohort_provenance.json"
  local manifest="experiments/brace/seeds/multitask_v1/${task}.json"
  if [[ ! -f "${prov}" || ! -f "${manifest}" ]]; then
    return 1
  fi
  local prov_manifest_sha prov_evidence_sha prov_config_sha
  prov_manifest_sha="$(python -c "import json;print(json.load(open('${prov}'))['manifest_sha256'])")"
  prov_evidence_sha="$(python -c "import json;print(json.load(open('${prov}'))['evidence_sha256'] or '')")"
  prov_config_sha="$(python -c "import json;print(json.load(open('${prov}'))['task_config_sha256'])")"
  [[ "${prov_manifest_sha}" == "$(manifest_sha256 "${manifest}")" ]] || return 1
  [[ "${prov_evidence_sha}" == "$(python -c "import json;m=json.load(open('${manifest}'));print((m.get('feasibility',{}) or {}).get('evidence_sha256') or '')")" ]] || return 1
  [[ "${prov_config_sha}" == "$(manifest_sha256 task_config/demo_clean.yml)" ]] || return 1
  local expected actual
  expected="$(expected_cohort_seeds "${task}")"
  actual="$(cat "data/${task}/demo_clean/seed.txt")"
  [[ "${expected}" == "${actual}" ]] || return 1
  return 0
}

quarantine_mismatched_demo_dir() {
  local task=$1
  local demo_dir="data/${task}/demo_clean"
  if [[ ! -d "${demo_dir}" ]]; then
    return 0
  fi
  local expected
  expected="$(expected_cohort_seeds "${task}" 2>/dev/null || true)"
  if [[ -z "${expected}" ]]; then
    echo "WARN ${task}: no frozen expert_demo cohort in manifest; skip quarantine" | tee -a "${manifest_log}"
    return 0
  fi
  local mismatched=0
  if [[ -f "${demo_dir}/seed.txt" ]]; then
    if [[ "$(cat "${demo_dir}/seed.txt")" != "${expected}" ]]; then
      mismatched=1
    fi
  elif [[ -d "${demo_dir}/data" || -d "${demo_dir}/_traj_data" ]]; then
    mismatched=1
  fi
  if [[ -f "${demo_dir}/cohort_provenance.json" ]] && ! cohort_matches_manifest "${task}"; then
    mismatched=1
  fi
  if (( mismatched == 1 )); then
    local legacy="data/${task}/demo_clean_legacy_$(date -u +%Y%m%dT%H%M%SZ)"
    echo "QUARANTINE ${task}: mismatched legacy demo dir -> ${legacy}" | tee -a "${manifest_log}"
    mkdir -p "data/${task}"
    mv "${demo_dir}" "${legacy}"
  fi
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

task_assets_ready() {
  local task=$1
  local ckpt="policy/DP/checkpoints/${task}-demo_clean-${required_episodes}-0/600.ckpt"
  local zarr="policy/DP/data/${task}-demo_clean-${required_episodes}.zarr"
  [[ -f "${ckpt}" && -d "${zarr}" ]] && cohort_matches_manifest "${task}"
}

collect_only() {
  local task=$1 gpu=$2
  local log="${log_root}/${run_stamp}_${task}_collect.log"
  {
    set -euo pipefail
    echo "=== collect ${task} gpu=${gpu} ==="
    quarantine_mismatched_demo_dir "${task}"
    python experiments/brace/prepare_multitask_demo_seeds.py "${task}" --count "${required_episodes}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    bash collect_data.sh "${task}" demo_clean "${gpu}"
    local hdf5_count
    hdf5_count="$(count_demo_hdf5 "${task}")"
    if (( hdf5_count < required_episodes )); then
      echo "expected ${required_episodes} demo HDF5 for ${task}, found ${hdf5_count}" >&2
      exit 1
    fi
    echo "=== collect ${task} complete hdf5=${hdf5_count} ==="
  } >>"${log}" 2>&1
}

process_only() {
  local task=$1
  local log="${log_root}/${run_stamp}_${task}_process.log"
  {
    set -euo pipefail
    echo "=== process ${task} ==="
    local hdf5_count
    hdf5_count="$(count_demo_hdf5 "${task}")"
    if (( hdf5_count < required_episodes )); then
      echo "expected ${required_episodes} demo HDF5 for ${task}, found ${hdf5_count}" >&2
      exit 1
    fi
    remove_empty_demo_zarr "${task}"
    (cd policy/DP && bash process_data.sh "${task}" demo_clean "${required_episodes}")
    echo "=== process ${task} complete ==="
  } >>"${log}" 2>&1
}

train_only() {
  local task=$1 gpu=$2
  local log="${log_root}/${run_stamp}_${task}_train.log"
  {
    set -euo pipefail
    echo "=== train ${task} gpu=${gpu} ==="
    export CUDA_VISIBLE_DEVICES="${gpu}"
    (cd policy/DP && bash train.sh "${task}" demo_clean "${required_episodes}" 0 14 "${gpu}")
    echo "=== train ${task} complete ==="
  } >>"${log}" 2>&1
}

launch_packed_wave() {
  local stage_fn=$1
  shift
  local -a queue=("$@")
  local idx=0 total=${#queue[@]}
  while (( idx < total )); do
    local -a pids=() names=() gpus_used=()
    for gpu in "${gpu_ids[@]}"; do
      local slot=0
      while (( slot < collect_workers_per_gpu && idx < total )); do
        local task="${queue[idx]}"
        if [[ "${stage_fn}" == "collect_only" ]]; then
          echo "LAUNCH collect ${task} gpu=${gpu}" | tee -a "${manifest_log}"
          collect_only "${task}" "${gpu}" &
        elif [[ "${stage_fn}" == "train_only" ]]; then
          echo "LAUNCH train ${task} gpu=${gpu}" | tee -a "${manifest_log}"
          train_only "${task}" "${gpu}" &
        else
          echo "unknown stage_fn ${stage_fn}" >&2
          return 1
        fi
        pids+=("$!")
        names+=("${task}")
        gpus_used+=("${gpu}")
        idx=$((idx + 1))
        slot=$((slot + 1))
      done
    done
    local i
    local wave_failed=0
    for i in "${!pids[@]}"; do
      if ! wait "${pids[i]}"; then
        echo "FAILED ${names[i]} (${stage_fn} gpu=${gpus_used[i]})" | tee -a "${manifest_log}"
        wave_failed=$((wave_failed + 1))
      else
        echo "OK ${names[i]} (${stage_fn})" | tee -a "${manifest_log}"
      fi
    done
    return "${wave_failed}"
  done
  return 0
}

trap restore_demo_clean EXIT
patch_demo_clean_for_multitask

task_collect_ready() {
  local task=$1
  local hdf5_count
  hdf5_count="$(count_demo_hdf5 "${task}")"
  (( hdf5_count >= required_episodes )) && cohort_matches_manifest "${task}"
}

needs_collect() {
  local task=$1
  local hdf5_count
  hdf5_count="$(count_demo_hdf5 "${task}")"
  if (( hdf5_count < required_episodes )); then
    return 0
  fi
  if ! cohort_matches_manifest "${task}"; then
    return 0
  fi
  return 1
}

task_process_ready() {
  local task=$1
  local zarr="policy/DP/data/${task}-demo_clean-${required_episodes}.zarr"
  [[ -d "${zarr}" ]] && python - <<PY
import sys, zarr
root = zarr.open("${zarr}", mode="r")
sys.exit(0 if "episode_ends" in root["meta"] else 1)
PY
}

failed=0
completed=0
pending_collect=()
pending_process=()
pending_train=()
for task in "${tasks[@]}"; do
  if task_assets_ready "${task}"; then
    echo "SKIP ${task}: checkpoint and zarr exist with matching cohort provenance" | tee -a "${manifest_log}"
    completed=$((completed + 1))
    continue
  fi
  if needs_collect "${task}"; then
    pending_collect+=("${task}")
  fi
  if task_collect_ready "${task}" && ! task_process_ready "${task}"; then
    pending_process+=("${task}")
  fi
  if task_process_ready "${task}"; then
    pending_train+=("${task}")
  fi
done

echo "schedule stage=${line_b_stage} pending=${#pending_collect[@]} gpus=${gpu_ids[*]} collect_workers_per_gpu=${collect_workers_per_gpu}" | tee -a "${manifest_log}"

run_collect_stage() {
  ((${#pending_collect[@]} == 0)) && return 0
  local wave_failed=0
  launch_packed_wave collect_only "${pending_collect[@]}" || wave_failed=$?
  return "${wave_failed}"
}

run_process_stage() {
  local task
  for task in "${pending_process[@]}"; do
    echo "LAUNCH process ${task}" | tee -a "${manifest_log}"
    if ! process_only "${task}"; then
      echo "FAILED ${task} (process)" | tee -a "${manifest_log}"
      return 1
    fi
    echo "OK ${task} (process)" | tee -a "${manifest_log}"
  done
}

run_train_stage() {
  ((${#pending_train[@]} == 0)) && return 0
  # DP training is exclusive: one trainer per GPU per wave.
  local -a queue=("${pending_train[@]}")
  local idx=0 total=${#queue[@]}
  while (( idx < total )); do
    local -a pids=() names=() assigned_gpus=()
    for gpu in "${gpu_ids[@]}"; do
      if (( idx >= total )); then
        break
      fi
      local task="${queue[idx]}"
      echo "LAUNCH train ${task} gpu=${gpu}" | tee -a "${manifest_log}"
      train_only "${task}" "${gpu}" &
      pids+=("$!")
      names+=("${task}")
      assigned_gpus+=("${gpu}")
      idx=$((idx + 1))
    done
    local i
    for i in "${!pids[@]}"; do
      if ! wait "${pids[i]}"; then
        echo "FAILED ${names[i]} (train gpu=${assigned_gpus[i]})" | tee -a "${manifest_log}"
        return 1
      fi
      echo "OK ${names[i]} (train)" | tee -a "${manifest_log}"
    done
  done
}

case "${line_b_stage}" in
  collect)
    run_collect_stage || failed=$((failed + $?))
    ;;
  process)
    run_process_stage || failed=$((failed + 1))
    ;;
  train)
    run_train_stage || failed=$((failed + 1))
    ;;
  all)
    run_collect_stage && collect_rc=0 || collect_rc=$?
    failed=$((failed + collect_rc))
    if (( collect_rc == 0 )); then
      run_process_stage || failed=$((failed + 1))
    fi
    if (( failed == 0 )); then
      run_train_stage || failed=$((failed + 1))
    fi
    ;;
  *)
    echo "unknown BRACE_LINE_B_STAGE=${line_b_stage} (use collect|process|train|all)" >&2
    exit 2
    ;;
esac

echo "line_b_assets done stage=${line_b_stage} completed=${completed}/${#tasks[@]} failed=${failed}" | tee -a "${manifest_log}"
if ((failed > 0)); then
  exit 1
fi
