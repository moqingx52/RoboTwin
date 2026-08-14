#!/bin/bash
# BRACE multitask v2 primary substrate: native success-first Base200 assets.
#
# Unlike the frozen v1 Line-B launcher, this entry does not preselect a cohort
# or require expert seed replay stability. Every task consumes its frozen
# ascending candidate stream, saves successful seed+trajectory pairs, and
# continues past failed candidates. Task failures are reported but do not make
# the overall launcher fail unless BRACE_FAIL_ON_TASK_ERROR=1.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

source /root/miniconda/etc/profile.d/conda.sh
conda activate RoboTwin
export PYTHONPATH="${repo_root}:${repo_root}/policy/DP${PYTHONPATH:+:${PYTHONPATH}}"

protocol=${BRACE_MULTITASK_V2_PROTOCOL:-experiments/brace/multitask_protocol.v2.json}
stage=${BRACE_BASE200_STAGE:-all}
target=${BRACE_EXPERT_TARGET_SUCCESSES:-200}
candidate_limit=${BRACE_CANDIDATE_SEED_LIMIT:-5000}
workers_per_gpu=${BRACE_BASE200_COLLECT_WORKERS_PER_GPU:-3}
fail_on_task_error=${BRACE_FAIL_ON_TASK_ERROR:-0}
read -r -a gpu_ids <<< "${BRACE_GPU_IDS:-0 1 2 3 4 5 6 7}"

protocol_sha="$(python - "${protocol}" <<'PY'
import hashlib, json, sys
from pathlib import Path
path = Path(sys.argv[1])
p = json.loads(path.read_text(encoding="utf-8"))
errors = []
if p.get("protocol_revision") != "brace.multitask.v2": errors.append("unexpected protocol revision")
if p.get("status") != "base_acquisition_frozen_method_pending": errors.append("base acquisition is not frozen")
base = p.get("base_dataset", {})
if base.get("target_successful_trajectories") != 200: errors.append("Base200 target must be 200")
if base.get("selection_rule") != "ascending_candidate_seed_success_first": errors.append("unexpected selection rule")
starts = list(p.get("task_candidate_seed_start", {}).values())
limit = int(base.get("candidate_seed_limit_per_task", 0))
if len(starts) != len(set(starts)) or any(abs(a-b) < limit for i,a in enumerate(starts) for b in starts[i+1:]):
    errors.append("candidate seed ranges overlap")
sidecar = path.with_suffix(path.suffix + ".sha256")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").split()[0] != digest:
    errors.append("missing or invalid protocol SHA256 sidecar")
if errors: raise SystemExit("invalid multitask v2 protocol: " + "; ".join(errors))
print(digest)
PY
)"
protocol_candidate_limit="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_dataset"]["candidate_seed_limit_per_task"])' "${protocol}")"
if [[ "${candidate_limit}" != "${protocol_candidate_limit}" ]]; then
  echo "BRACE_CANDIDATE_SEED_LIMIT=${candidate_limit} conflicts with frozen protocol value ${protocol_candidate_limit}" >&2
  exit 2
fi

mapfile -t protocol_tasks < <(python - "${protocol}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
print(p["development_task"])
print(*p["heldout_tasks"], sep="\n")
PY
)
if [[ -n "${BRACE_BASE200_TASKS:-}" ]]; then
  read -r -a tasks <<< "${BRACE_BASE200_TASKS}"
else
  tasks=("${protocol_tasks[@]}")
fi

if ((${#gpu_ids[@]} == 0 || workers_per_gpu < 1)); then
  echo "GPU list must be non-empty and workers-per-GPU positive" >&2
  exit 2
fi
if [[ "${target}" != "200" ]]; then
  echo "multitask v2 primary acquisition requires exactly 200 successes" >&2
  exit 2
fi
case "${stage}" in collect|process|train|all) ;; *) echo "use BRACE_BASE200_STAGE=collect|process|train|all" >&2; exit 2 ;; esac

run_dir=${BRACE_BASE200_RUN_DIR:-experiments/brace/runs/base200_assets_$(date -u +%Y%m%dT%H%M%SZ)}
log_root=${BRACE_BASE200_LOG_DIR:-experiments/brace/logs/base200_assets}
mkdir -p "${run_dir}" "${log_root}"
run_id="$(basename "${run_dir}")"
manifest_log="${log_root}/${run_id}_manifest.log"
git_commit="$(git rev-parse HEAD)"

python experiments/brace/write_run_meta.py \
  --run-dir "${run_dir}" --stage "base200_assets_${stage}" --run-id "${run_id}" \
  --tasks "${tasks[@]}" --extra \
  "protocol=${protocol}" "target_successes=${target}" "candidate_seed_limit=${candidate_limit}" \
  "gpu_ids=${gpu_ids[*]}" "collect_workers_per_gpu=${workers_per_gpu}" \
  "task_failure_policy=fail_open" "official_base50_role=sensitivity_only"

seed_start() {
  python - "${protocol}" "$1" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
task = sys.argv[2]
if task not in p["task_candidate_seed_start"]:
    raise SystemExit(f"task not registered in multitask v2: {task}")
print(p["task_candidate_seed_start"][task])
PY
}

hdf5_count() {
  local dir="data/$1/demo_clean/data"
  [[ -d "${dir}" ]] || { echo 0; return; }
  find "${dir}" -maxdepth 1 -type f -name 'episode*.hdf5' | wc -l | tr -d ' '
}

seed_count() {
  local path="data/$1/demo_clean/seed.txt"
  [[ -f "${path}" ]] || { echo 0; return; }
  python - "${path}" <<'PY'
import sys
print(len(open(sys.argv[1], encoding="utf-8").read().split()))
PY
}

acquisition_matches() {
  local task=$1 log="data/$1/demo_clean/expert_acquisition_attempts.jsonl" start
  [[ -s "${log}" ]] || return 1
  start="$(seed_start "${task}")"
  python - "${log}" "${task}" "${start}" "${target}" <<'PY'
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1], encoding="utf-8") if x.strip()]
ok = bool(rows) and all(
    r.get("schema_version") == 2
    and r.get("task") == sys.argv[2]
    and r.get("candidate_seed_start") == int(sys.argv[3])
    and r.get("target_successes") == int(sys.argv[4])
    for r in rows
)
raise SystemExit(0 if ok else 1)
PY
}

preserve_non_v2_raw_data() {
  local task=$1 dir="data/$1/demo_clean"
  [[ -d "${dir}" ]] || return 0
  acquisition_matches "${task}" && return 0
  local archive="data/${task}/demo_clean_pre_v2_$(date -u +%Y%m%dT%H%M%SZ)"
  echo "PRESERVE ${task}: moving non-v2 raw data to ${archive}" | tee -a "${manifest_log}"
  mv "${dir}" "${archive}"
}

raw_ready() {
  local task=$1
  [[ "$(hdf5_count "${task}")" == "${target}" ]] \
    && [[ "$(seed_count "${task}")" == "${target}" ]] \
    && acquisition_matches "${task}"
}

zarr_ready() {
  local zarr="policy/DP/data/$1-demo_clean-${target}.zarr"
  [[ -d "${zarr}" ]] && python - "${zarr}" "$1" "${protocol_sha}" <<'PY'
import json, sys, zarr
r = zarr.open(sys.argv[1], mode="r")
prov_path = sys.argv[1] + "/brace_base200_provenance.json"
try: prov = json.load(open(prov_path, encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError): prov = {}
ok = (
    "meta" in r and "episode_ends" in r["meta"] and len(r["meta"]["episode_ends"]) == 200
    and prov.get("task") == sys.argv[2] and prov.get("protocol_sha256") == sys.argv[3]
)
raise SystemExit(0 if ok else 1)
PY
}

checkpoint_ready() {
  local dir="policy/DP/checkpoints/$1-demo_clean-${target}-0"
  [[ -f "${dir}/600.ckpt" && -f "${dir}/brace_base200_provenance.json" ]] \
    && python - "${dir}/brace_base200_provenance.json" "$1" "${protocol_sha}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if p.get("task") == sys.argv[2] and p.get("protocol_sha256") == sys.argv[3] else 1)
PY
}

collect_task() {
  local task=$1 gpu=$2 start log="${log_root}/${run_id}_$1_collect.log"
  start="$(seed_start "${task}")"
  {
    preserve_non_v2_raw_data "${task}"
    if raw_ready "${task}"; then echo "SKIP raw ready ${task}"; return 0; fi
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export BRACE_SUCCESS_FIRST=1 BRACE_EXPERT_TARGET_SUCCESSES="${target}"
    export BRACE_CANDIDATE_SEED_START="${start}" BRACE_CANDIDATE_SEED_LIMIT="${candidate_limit}"
    export BRACE_GIT_COMMIT="${git_commit}"
    unset BRACE_PREMOTION_MAX_ATTEMPTS BRACE_PREMOTION_RETRY_AMENDMENT
    bash collect_data.sh "${task}" demo_clean "${gpu}"
    raw_ready "${task}"
  } >>"${log}" 2>&1
}

process_task() {
  local task=$1 log="${log_root}/${run_id}_$1_process.log"
  {
    raw_ready "${task}"
    local zarr="policy/DP/data/${task}-demo_clean-${target}.zarr"
    if [[ -d "${zarr}" ]] && ! zarr_ready "${task}"; then
      mv "${zarr}" "${zarr}.pre_v2_$(date -u +%Y%m%dT%H%M%SZ)"
    fi
    (cd policy/DP && bash process_data.sh "${task}" demo_clean "${target}")
    python - "${zarr}" "${task}" "${protocol}" "${protocol_sha}" "${git_commit}" <<'PY'
import hashlib, json, sys
from pathlib import Path
zarr, task, protocol, protocol_sha, commit = sys.argv[1:]
raw = Path("data") / task / "demo_clean"
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
p = {"schema_version": 1, "task": task, "target_successes": 200, "protocol": protocol,
     "protocol_sha256": protocol_sha, "git_commit": commit,
     "seed_txt_sha256": sha(raw / "seed.txt"),
     "attempt_log_sha256": sha(raw / "expert_acquisition_attempts.jsonl")}
Path(zarr, "brace_base200_provenance.json").write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")
PY
    zarr_ready "${task}"
  } >>"${log}" 2>&1
}

train_task() {
  local task=$1 gpu=$2 log="${log_root}/${run_id}_$1_train.log"
  {
    zarr_ready "${task}"
    local ckpt_dir="policy/DP/checkpoints/${task}-demo_clean-${target}-0"
    if [[ -d "${ckpt_dir}" ]] && ! checkpoint_ready "${task}"; then
      mv "${ckpt_dir}" "${ckpt_dir}.pre_v2_$(date -u +%Y%m%dT%H%M%SZ)"
    fi
    export CUDA_VISIBLE_DEVICES="${gpu}"
    (cd policy/DP && bash train.sh "${task}" demo_clean "${target}" 0 14 "${gpu}")
    [[ -f "${ckpt_dir}/600.ckpt" ]]
    python - "${ckpt_dir}" "${task}" "${protocol}" "${protocol_sha}" "${git_commit}" <<'PY'
import json, sys
from pathlib import Path
directory, task, protocol, protocol_sha, commit = sys.argv[1:]
p = {"schema_version": 1, "task": task, "target_successes": 200, "protocol": protocol,
     "protocol_sha256": protocol_sha, "git_commit": commit}
Path(directory, "brace_base200_provenance.json").write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")
PY
    checkpoint_ready "${task}"
  } >>"${log}" 2>&1
}

run_gpu_wave() {
  local fn=$1 per_gpu=$2; shift 2
  local -a queue=("$@") pids=() names=() assigned=()
  local idx=0 failures=0
  while ((idx < ${#queue[@]})); do
    pids=(); names=(); assigned=()
    for gpu in "${gpu_ids[@]}"; do
      for ((slot=0; slot<per_gpu && idx<${#queue[@]}; slot++)); do
        local task="${queue[idx]}"
        echo "LAUNCH ${fn} ${task} gpu=${gpu}" | tee -a "${manifest_log}"
        "${fn}" "${task}" "${gpu}" &
        pids+=("$!"); names+=("${task}"); assigned+=("${gpu}"); idx=$((idx + 1))
      done
    done
    for i in "${!pids[@]}"; do
      if wait "${pids[i]}"; then
        echo "OK ${fn} ${names[i]}" | tee -a "${manifest_log}"
      else
        echo "FAILED ${fn} ${names[i]} gpu=${assigned[i]}" | tee -a "${manifest_log}"
        failures=$((failures + 1))
      fi
    done
  done
  return "${failures}"
}

collect_failures=0 process_failures=0 train_failures=0
if [[ "${stage}" == collect || "${stage}" == all ]]; then
  run_gpu_wave collect_task "${workers_per_gpu}" "${tasks[@]}" || collect_failures=$?
fi

if [[ "${stage}" == process || "${stage}" == all ]]; then
  for task in "${tasks[@]}"; do
    raw_ready "${task}" || continue
    zarr_ready "${task}" && continue
    echo "LAUNCH process ${task}" | tee -a "${manifest_log}"
    if process_task "${task}"; then echo "OK process ${task}" | tee -a "${manifest_log}"; else process_failures=$((process_failures + 1)); echo "FAILED process ${task}" | tee -a "${manifest_log}"; fi
  done
fi

if [[ "${stage}" == train || "${stage}" == all ]]; then
  train_queue=()
  for task in "${tasks[@]}"; do zarr_ready "${task}" && ! checkpoint_ready "${task}" && train_queue+=("${task}"); done
  ((${#train_queue[@]} == 0)) || run_gpu_wave train_task 1 "${train_queue[@]}" || train_failures=$?
fi

python - "${run_dir}" "${protocol}" "${stage}" "${collect_failures}" "${process_failures}" "${train_failures}" "${tasks[@]}" <<'PY'
import json, sys
from pathlib import Path
run, protocol, stage = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
cf, pf, tf = map(int, sys.argv[4:7])
tasks = sys.argv[7:]
def count(task):
    return len(list((Path("data") / task / "demo_clean" / "data").glob("episode*.hdf5")))
rows = []
for task in tasks:
    raw = count(task)
    zarr = Path(f"policy/DP/data/{task}-demo_clean-200.zarr").is_dir()
    ckpt = Path(f"policy/DP/checkpoints/{task}-demo_clean-200-0/600.ckpt").is_file()
    rows.append({"task": task, "raw_hdf5": raw, "raw_ready": raw == 200, "zarr_ready": zarr, "checkpoint_ready": ckpt})
payload = {"schema_version": 1, "stage": stage, "protocol": protocol, "task_failure_policy": "fail_open", "stage_failures": {"collect": cf, "process": pf, "train": tf}, "tasks": rows}
tmp = run / "summary.json.tmp"
tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
tmp.replace(run / "summary.json")
PY

total_failures=$((collect_failures + process_failures + train_failures))
echo "base200 assets complete stage=${stage} failures=${total_failures} run_dir=${run_dir}" | tee -a "${manifest_log}"
if [[ "${fail_on_task_error}" == 1 && "${total_failures}" != 0 ]]; then exit 1; fi
exit 0
