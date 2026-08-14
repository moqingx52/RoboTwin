#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

stage=${1:-help}
shift || true

read -r -a tasks <<< "${BRACE_TASKS:-place_container_plate dump_bin_bigbin}"
read -r -a gpu_ids <<< "${BRACE_GPU_IDS:-0 1 2 3 4 5 6 7}"

rollout_dir=${BRACE_ROLLOUT_DIR:-experiments/phase1/rollouts_200}
traced_rollout_dir=${BRACE_TRACED_ROLLOUT_DIR:-experiments/brace/rollouts_traced}
brace_dir=${BRACE_OUTPUT_DIR:-experiments/brace}
protocol=${BRACE_PROTOCOL_PATH:-${brace_dir}/protocol.json}
protocol_v2=${BRACE_PROTOCOL_V2_PATH:-experiments/brace/protocol.v2.3.json}
pilot_rollout_dir=${BRACE_PILOT_ROLLOUT_DIR:-experiments/brace/rollouts_traced_pilot}
num_shards=${BRACE_NUM_SHARDS:-12}
rollouts_per_seed=${BRACE_ROLLOUTS_PER_SEED:-8}
verify_workers=${BRACE_VERIFY_WORKERS:-96}
audit_workers_per_gpu=${BRACE_AUDIT_WORKERS_PER_GPU:-3}
audit_workers=${BRACE_AUDIT_WORKERS:-$(( ${#gpu_ids[@]} * audit_workers_per_gpu ))}
branch_prepare_workers=${BRACE_BRANCH_PREPARE_WORKERS:-96}
branch_output_dir=${BRACE_BRANCH_OUTPUT_DIR:-${brace_dir}/branches}
dataset_dir=${BRACE_DATASET_DIR:-${brace_dir}/datasets}
dataset_run_label=${BRACE_DATASET_RUN_LABEL:-place_pilot_v2.3}
export_protocol=${BRACE_EXPORT_PROTOCOL_PATH:-${brace_dir}/export_protocol.v1.2.json}
screen_protocol=${BRACE_SCREEN_PROTOCOL_PATH:-${brace_dir}/screen_protocol.v1.2.json}
confirmatory_protocol=${BRACE_CONFIRMATORY_PROTOCOL_PATH:-${brace_dir}/screen_protocol.v1.4.2.confirmatory_preservation.json}
confirmatory_seeds_file=${BRACE_CONFIRMATORY_SEEDS_FILE:-${brace_dir}/seeds/place_container_plate_confirmatory_v1.4.2_seeds.json}
confirmatory_jobs=${BRACE_CONFIRMATORY_JOBS:-${brace_dir}/confirmatory_preservation_jobs.place_container_plate.v3.json}
multitask_protocol=${BRACE_MULTITASK_PROTOCOL_PATH:-${brace_dir}/multitask_protocol.v1.json}

# shellcheck source=experiments/brace/run_paths.sh
source "${repo_root}/experiments/brace/run_paths.sh"
export BRACE_RUN_ID="${BRACE_RUN_ID:-$(brace_utc_run_id)}"

require_v12_developmental_pass() {
  local promotion="${brace_dir}/promotions/screen_v1.2.json"
  if [[ ! -f "${promotion}" ]]; then
    echo "Blocked: no passing screen.v1.2 developmental promotion at ${promotion}" >&2
    exit 2
  fi
  if [[ "$(jq -r '.passed // false' "${promotion}")" != "true" ]]; then
    echo "Blocked: screen.v1.2 developmental screen has not passed: ${promotion}" >&2
    exit 2
  fi
}

block_dump_screen_until_place_pass() {
  local revision
  revision="$(jq -r '.protocol_revision // ""' "${screen_protocol}")"
  if [[ "${revision}" != "screen.v1.2" ]]; then
    return 0
  fi
  for task in "${tasks[@]}"; do
    if [[ "${task}" == "dump_bin_bigbin" ]]; then
      require_v12_developmental_pass
    fi
  done
}

pilot_seeds_file_for_task() {
  local task=$1
  local seeds_path
  if [[ -n "${BRACE_PILOT_SEEDS_FILE:-}" ]]; then
    echo "${BRACE_PILOT_SEEDS_FILE}"
  elif seeds_path="$(brace_resolve_seeds_file "${task}_pilot_seeds.json" 2>/dev/null)"; then
    echo "${seeds_path}"
  else
    echo "${brace_dir}/seeds/${task}_pilot_seeds.json"
  fi
}

pilot_shard_count_for_task() {
  local task=$1
  local seeds_file seed_count max_workers
  seeds_file="$(pilot_seeds_file_for_task "${task}")"
  if [[ -n "${BRACE_PILOT_NUM_SHARDS:-}" ]]; then
    echo "${BRACE_PILOT_NUM_SHARDS}"
    return
  fi
  seed_count="$(jq '.seeds | length' "${seeds_file}")"
  max_workers=$(( ${#gpu_ids[@]} * ${BRACE_ROLLOUT_WORKERS_PER_GPU:-3} / ${#tasks[@]} ))
  if (( seed_count > 0 && max_workers > seed_count )); then
    echo "${seed_count}"
  else
    echo "${max_workers}"
  fi
}

usage() {
  cat <<'EOF'
BRACE unified experiment entry

Usage:
  bash experiments/brace/run_all.sh <stage>

Stages:
  status   Read-only progress summary for the currently running rollout collection.
  collect  Start/resume failure-HDF5 collection through the existing Phase 3 collector.
  verify   Strictly verify all shards and rebuild canonical manifests.
  init     Create BRACE directories and a local protocol.json from the template.
  audit    Run v1 waypoint-replay audit (historical baseline).
  audit-v2 Run snapshot + control-trace audit (requires protocol.v2.3.json).
  collect-trace-smoke  Collect 2-4 traced rollouts per task for schema smoke.
  collect-trace-audit  Collect traced rollouts for the v2 audit sample.
  select-pilot-seeds  Select mixed-outcome env seeds for Stage-2 pilot.
  select-confirm-seeds  Select held-out confirmatory seeds (Track B).
  collect-trace-pilot  Collect traced rollouts for Stage-2 pilot seeds.
  verify-traced        Verify traced rollout shards and schema v2 HDF5.
  inventory-traced     Inventory mixed-outcome seed counts (directed collection).
  artifact-inventory   Scan evidence bundles; write sync inventory + checklist (cloud-friendly).
  validate-artifacts   Validate archives/manifests against optional remote inventory snapshot.
  merge-audit-v2       Merge per-task replay audit summaries into combined gate file.
  archive-replay-gate  Extract per-task replay gate summary into archive/.
  promote-run          Promote immutable run outputs into archive/ with source_run.json.
  audit-mutable-paths  Static scan for writes to deprecated mutable JSON paths.
  list-runs            List recent timestamped BRACE run directories.
  list-records         List recent stage metadata records (experiments/brace/records/).
  export-verified-chunks  Export B1/N1 chunk manifests from branch artifacts.
  prepare-hard-seeds   Base ID probe + select held-out hard eval seeds for screen.
  select-preservation-cohort  Build disjoint confirmatory preservation/boundary cohorts.
  confirmatory-base-census  P1a: base census at offset 3000 (ID+train only).
  confirmatory-preservation   P1c: frozen A1 vs SFT-only (5 training seeds, 10 jobs).
  confirmatory-preservation-eval  P1d: fresh paired eval at offset 4000.
  confirmatory-preservation-report  Aggregate H1 gates from P1d eval artifacts.
  multitask-validate  Validate the frozen 12-task design (method freeze optional).
  multitask-generate-seeds  Generate deterministic disjoint candidate seed manifests.
  multitask-scan-feasibility  Scan rollout_train solvability; select expert_demo cohort.
  multitask-line-b-assets  Collect/process/train 50-demo assets from frozen expert_demo.
  multitask-base200-assets Collect/process/train v2 Base200 with native success-first seeds.
  multitask-preflight  Audit 50-demo DP, seed, replay, and method-freeze readiness.
  multitask-run  Run/resume a frozen explicit multitask job manifest.
  multitask-report  Aggregate the complete held-out artifact matrix.
  branch   Collect matched-continuation branches (requires passed replay audit v2).
  screen   Run B1/B2/B3/N1 screen (requires branch gate).
  full     Run preregistered Base/U1/U4/B1/B2/B3 full evaluation.

Important:
  The old experiments/phase3 prep/screen/full stages are not called by this entry.
  The currently running failure collection is useful BRACE input and should finish.

Environment (v2 audit / branch):
  BRACE_PROTOCOL_V2_PATH   Protocol file for audit-v2 and branch (default:
                           experiments/brace/protocol.v2.3.json). Set explicitly when
                           reproducing archived v2.3 runs.
  BRACE_TRACED_ROLLOUT_DIR Traced HDF5 root. Use rollouts_traced for anchor/screen;
                           rollouts_traced_pilot is for Stage-2 branch collection only.
  BRACE_AUDIT_RUN_DIR      Explicit replay-audit run dir; must have replay_gate_passed=true.
  BRACE_BRANCH_OUTPUT_DIR  Branch summary/checks output dir (default: experiments/brace/branches).
  BRACE_PILOT_SEEDS_FILE   Override pilot/confirm seeds JSON for collect/verify/branch.
  BRACE_HARD_EVAL_DIR      Output root for base probe + hard seeds (default: eval_results_200).
  BRACE_HARD_PROBE_SHARDS  Shard count for base ID probe (default: 8).
  BRACE_FORCE_HARD_SEEDS   Set to 1 to regenerate existing hard_eval_seeds files.
  BRACE_TASKS              Space-separated task subset (default: both protocol tasks).
  BRACE_MULTITASK_PROTOCOL_PATH  Multitask protocol path.
  BRACE_BASE200_STAGE       collect|process|train|all (default all).
  BRACE_BASE200_TASKS       Optional v2 task subset; defaults to development + 10 held-out.
  BRACE_FAIL_ON_TASK_ERROR  Set 1 for operator-facing nonzero exit; default 0 is task-local fail-open.

Immutable outputs (default since v2.3+):
  Each stage writes under experiments/brace/runs/<UTC>_<stage>_<tasks>/...
  Pointer files: runs/LATEST, runs/LATEST_AUDIT_<task>, runs/LATEST_<branch_label>
  Set BRACE_LEGACY_MUTABLE_OUTPUTS=1 to restore old shared paths (not recommended).
  Frozen git-tracked copies live under experiments/brace/archive/ only.
EOF
}

collection_status() {
  local expected_per_task=0
  local seed_file task task_dir shard_rows success_files failure_files

  printf "rollout_dir=%s\n" "${rollout_dir}"
  for task in "${tasks[@]}"; do
    seed_file="experiments/phase1/seeds/${task}_seeds.json"
    if [[ -s "${seed_file}" ]]; then
      expected_per_task=$(( $(jq '.train_rollout | length' "${seed_file}") * rollouts_per_seed ))
    else
      expected_per_task=0
    fi

    task_dir="${rollout_dir}/${task}"
    shard_rows=0
    if [[ -d "${task_dir}" ]]; then
      while IFS= read -r -d '' manifest; do
        shard_rows=$(( shard_rows + $(wc -l < "${manifest}") ))
      done < <(find "${task_dir}" -maxdepth 1 -type f \
        -name "manifest_shard_*_of_$(printf '%02d' "${num_shards}").jsonl" -print0)
    fi

    success_files=0
    failure_files=0
    if [[ -d "${task_dir}/successes" ]]; then
      success_files=$(find "${task_dir}/successes" -maxdepth 1 -type f -name '*.hdf5' | wc -l)
    fi
    if [[ -d "${task_dir}/failures" ]]; then
      failure_files=$(find "${task_dir}/failures" -maxdepth 1 -type f -name '*.hdf5' | wc -l)
    fi
    printf "%s rows=%d/%d successes=%d failures=%d\n" \
      "${task}" "${shard_rows}" "${expected_per_task}" "${success_files}" "${failure_files}"
  done

  if pgrep -af "experiments/phase1/collect_rollouts.py.*--save-failures" >/dev/null 2>&1; then
    echo "workers=running"
    pgrep -af "experiments/phase1/collect_rollouts.py.*--save-failures" | sed -n '1,6p'
  else
    echo "workers=not-running"
  fi
}

freeze_guard() {
  if [[ ! -s "${protocol}" ]]; then
    echo "Missing ${protocol}. Run: bash experiments/brace/run_all.sh init" >&2
    exit 2
  fi
  if rg -q "FREEZE_BEFORE_RUN|template_not_frozen" "${protocol}"; then
    echo "Protocol is not frozen: ${protocol}" >&2
    echo "Fill replay tolerances, set status to frozen, and archive the file before running experiments." >&2
    exit 2
  fi
}

require_gate() {
  local path=$1
  local message=$2
  if [[ ! -s "${path}" ]] || [[ "$(jq -r '.passed // false' "${path}")" != "true" ]]; then
    echo "${message}: ${path}" >&2
    exit 2
  fi
}

require_task_replay_gates() {
  local task task_summary
  for task in "${tasks[@]}"; do
    if ! task_summary="$(brace_latest_audit_summary "${task}")"; then
      echo "Replay audit v2 per-task gate has not passed for ${task}: no summary found" >&2
      exit 2
    fi
    if [[ "$(jq -r '.tasks[$task].replay_gate_passed // false' --arg task "${task}" "${task_summary}")" != "true" ]]; then
      echo "Replay audit v2 per-task gate has not passed for ${task}: ${task_summary}" >&2
      exit 2
    fi
  done
}

run_hard_probe_sharded() {
  local task=$1
  local ckpt_path=$2
  local probe_dir=$3
  local hard_shards=${BRACE_HARD_PROBE_SHARDS:-8}
  local failed=0
  local next_shard=0
  local running=0
  local worker_count=${#gpu_ids[@]}
  local -a worker_pids=()
  local slot shard gpu

  for ((slot=0; slot<worker_count; slot++)); do
    worker_pids[slot]=0
  done

  while (( next_shard < hard_shards || running > 0 )); do
    for ((slot=0; slot<worker_count; slot++)); do
      if (( worker_pids[slot] != 0 )) && ! kill -0 "${worker_pids[slot]}" 2>/dev/null; then
        wait "${worker_pids[slot]}" || failed=1
        worker_pids[slot]=0
        running=$((running - 1))
      fi
    done

    for ((slot=0; slot<worker_count; slot++)); do
      if (( next_shard < hard_shards && worker_pids[slot] == 0 )); then
        shard=${next_shard}
        gpu=${gpu_ids[slot]}
        (
          export CUDA_VISIBLE_DEVICES="${gpu}"
          python experiments/brace/prepare_hard_seeds.py \
            --task "${task}" \
            --ckpt-path "${ckpt_path}" \
            --probe-dir "${probe_dir}" \
            --hard-output "${probe_dir}/.unused_hard.json" \
            --num-shards "${hard_shards}" \
            --shard-id "${shard}" \
            --resume
        ) >"${brace_dir}/logs/prepare_hard_probe_${task}_shard${shard}.log" 2>&1 &
        worker_pids[slot]=$!
        next_shard=$((next_shard + 1))
        running=$((running + 1))
      fi
    done

    if (( running > 0 )); then
      sleep 1
    fi
  done

  if (( failed != 0 )); then
    echo "One or more hard-probe shards failed for ${task}. Inspect ${brace_dir}/logs/." >&2
    return 1
  fi
}

case "${stage}" in
  multitask-validate)
    validation_args=(validate --protocol "${multitask_protocol}")
    if [[ "${BRACE_REQUIRE_METHOD_FREEZE:-0}" == "1" ]]; then
      validation_args+=(--require-method-freeze)
    fi
    exec python experiments/brace/multitask_protocol.py "${validation_args[@]}" "$@"
    ;;

  multitask-generate-seeds)
    exec python experiments/brace/multitask_protocol.py generate-seeds \
      --protocol "${multitask_protocol}" "$@"
    ;;

  multitask-scan-feasibility)
    export BRACE_GPU_IDS="${BRACE_GPU_IDS:-0 1 2 3 4 5 6 7}"
    exec bash experiments/brace/run_multitask_feasibility_scan.sh "$@"
    ;;

  multitask-line-b-assets)
    export BRACE_GPU_IDS="${BRACE_GPU_IDS:-0 1 2 3 4 5 6 7}"
    exec bash experiments/brace/run_multitask_line_b_assets.sh "$@"
    ;;

  multitask-base200-assets)
    export BRACE_GPU_IDS="${BRACE_GPU_IDS:-0 1 2 3 4 5 6 7}"
    exec bash experiments/brace/run_multitask_base200_assets.sh "$@"
    ;;

  multitask-preflight)
    preflight_args=(--protocol "${multitask_protocol}")
    if [[ "${BRACE_REQUIRE_METHOD_FREEZE:-1}" == "1" ]]; then
      preflight_args+=(--require-method-freeze)
    fi
    exec python experiments/brace/multitask_preflight.py "${preflight_args[@]}" "$@"
    ;;

  multitask-run)
    if [[ -z "${BRACE_MULTITASK_JOBS:-}" || -z "${BRACE_MULTITASK_RUN_DIR:-}" ]]; then
      echo "Set BRACE_MULTITASK_JOBS and BRACE_MULTITASK_RUN_DIR." >&2
      exit 2
    fi
    exec python experiments/brace/multitask_scheduler.py \
      --protocol "${multitask_protocol}" \
      --jobs "${BRACE_MULTITASK_JOBS}" \
      --output-dir "${BRACE_MULTITASK_RUN_DIR}" \
      --gpus "${gpu_ids[@]}" "$@"
    ;;

  multitask-report)
    if [[ -z "${BRACE_MULTITASK_ARTIFACT_INDEX:-}" || -z "${BRACE_MULTITASK_REPORT_OUTPUT:-}" ]]; then
      echo "Set BRACE_MULTITASK_ARTIFACT_INDEX and BRACE_MULTITASK_REPORT_OUTPUT." >&2
      exit 2
    fi
    exec python experiments/brace/aggregate_multitask.py \
      --protocol "${multitask_protocol}" \
      --artifact-index "${BRACE_MULTITASK_ARTIFACT_INDEX}" \
      --output "${BRACE_MULTITASK_REPORT_OUTPUT}" "$@"
    ;;

  help|-h|--help)
    usage
    ;;

  status)
    collection_status
    ;;

  collect)
    if pgrep -af "experiments/phase1/collect_rollouts.py.*--save-failures" >/dev/null 2>&1; then
      echo "Failure collection workers are already running; refusing to launch duplicates." >&2
      collection_status
      exit 2
    fi
    export PHASE3_TASKS="${BRACE_TASKS:-place_container_plate dump_bin_bigbin}"
    export PHASE3_GPU_IDS="${BRACE_GPU_IDS:-0 1 2 3 4 5 6 7}"
    export PHASE3_ROLLOUT_WORKERS_PER_GPU="${BRACE_ROLLOUT_WORKERS_PER_GPU:-3}"
    export PHASE3_ROLLOUTS_PER_SEED="${rollouts_per_seed}"
    export PHASE3_ROLLOUT_DIR="${rollout_dir}"
    exec bash experiments/phase3/collect_failures_parallel.sh "$@"
    ;;

  verify)
    if pgrep -af "experiments/phase1/collect_rollouts.py.*--save-failures" >/dev/null 2>&1; then
      echo "Collection is still running. Wait for all workers before strict verification." >&2
      exit 2
    fi
    for task in "${tasks[@]}"; do
      python experiments/phase1/verify_rollouts.py \
        --tasks "${task}" \
        --rollout-dir "${rollout_dir}" \
        --rollouts-per-seed "${rollouts_per_seed}" \
        --num-shards "${num_shards}" \
        --require-failures

      python experiments/phase1/merge_rollout_shards.py \
        --task "${task}" \
        --task-config demo_clean \
        --seeds-file "experiments/phase1/seeds/${task}_seeds.json" \
        --rollouts-per-seed "${rollouts_per_seed}" \
        --rollout-dir "${rollout_dir}"
    done
    echo "BRACE input rollouts verified and canonical manifests rebuilt."
    ;;

  init)
    mkdir -p \
      "${brace_dir}/logs" \
      "${brace_dir}/runs" \
      "${brace_dir}/records" \
      "${brace_dir}/replay_audit" \
      "${brace_dir}/branches" \
      "${brace_dir}/budgets" \
      "${brace_dir}/run_states"
    if [[ ! -e "${protocol}" ]]; then
      cp experiments/brace/protocol.template.json "${protocol}"
      echo "Created ${protocol}"
    else
      echo "Preserved existing ${protocol}"
    fi
    echo "Next: freeze replay tolerances in ${protocol}, then run the audit stage."
    ;;

  audit)
    freeze_guard
    if [[ ! -f experiments/brace/replay_audit.py ]]; then
      echo "replay_audit.py is not implemented yet; audit cannot be claimed or bypassed." >&2
      exit 2
    fi
    exec python experiments/brace/replay_audit.py \
      --protocol "${protocol}" \
      --rollout-dir "${rollout_dir}" \
      --output-dir "${brace_dir}/replay_audit" \
      --gpus "${gpu_ids[@]}" \
      "$@"
    ;;

  audit-v2)
    if [[ ! -s "${protocol_v2}" ]]; then
      echo "Missing ${protocol_v2}" >&2
      exit 2
    fi
    if rg -q "FREEZE_BEFORE_RUN|template_not_frozen" "${protocol_v2}"; then
      echo "Protocol v2 is not frozen: ${protocol_v2}" >&2
      exit 2
    fi
    if [[ ! -f experiments/brace/replay_audit_v2.py ]]; then
      echo "replay_audit_v2.py is not implemented yet." >&2
      exit 2
    fi
    if [[ "${BRACE_LEGACY_MUTABLE_OUTPUTS:-0}" != "1" && -z "${BRACE_RUN_ROOT:-}" ]]; then
      BRACE_RUN_ROOT="$(brace_allocate_run_dir audit_v2 replay_audit_v2)"
      export BRACE_RUN_ROOT
    fi
    for task in "${tasks[@]}"; do
      if [[ "${BRACE_LEGACY_MUTABLE_OUTPUTS:-0}" == "1" && -z "${BRACE_AUDIT_OUTPUT_DIR:-}" ]]; then
        audit_output_dir="${brace_dir}/replay_audit_v2"
      else
        audit_output_dir="$(brace_audit_output_dir "${task}")"
      fi
      echo "Immutable audit output: ${audit_output_dir}" >&2
      python experiments/brace/replay_audit_v2.py \
        --protocol "${protocol_v2}" \
        --rollout-dir "${traced_rollout_dir}" \
        --output-dir "${audit_output_dir}" \
        --tasks "${task}" \
        --workers "${audit_workers}" \
        --workers-per-gpu "${audit_workers_per_gpu}" \
        --gpus "${gpu_ids[@]}" \
        "$@"
      if [[ "${BRACE_AUTO_PROMOTE:-1}" == "1" ]]; then
        if [[ -n "${BRACE_AUDIT_PROMOTE_TARGET:-}" ]]; then
          audit_target="${BRACE_AUDIT_PROMOTE_TARGET}"
        elif [[ "${traced_rollout_dir}" == *"rollouts_traced_base200"* ]]; then
          # Never overwrite the frozen place v2.3 pilot gate archive.
          audit_target=archive/replay_audit_v2_place_base200_v2_gate
        else
          audit_target=archive/replay_audit_v2_place_v2.3_gate
        fi
        if [[ "${task}" == "dump_bin_bigbin" ]]; then
          audit_target=archive/replay_audit_v2_dump_v2.3_gate
        fi
        BRACE_PROMOTE_RUN="${audit_output_dir}" \
        BRACE_PROMOTE_TARGET="${audit_target}" \
          bash experiments/brace/run_all.sh promote-run
      fi
    done
    ;;

  list-runs)
    if [[ ! -d "${brace_dir}/runs" ]]; then
      echo "No runs/ directory yet."
      exit 0
    fi
    find "${brace_dir}/runs" -maxdepth 2 -name meta.json -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | head -n "${BRACE_LIST_RUNS_LIMIT:-20}" \
      | while read -r _ts meta_path; do
          run_dir="$(dirname "${meta_path}")"
          echo "=== ${run_dir} ==="
          jq -c '{stage, tasks, created_at, git_commit}' "${meta_path}" 2>/dev/null || true
        done
    ;;

  list-records)
    python - <<'PY'
import json
from experiments.brace.stage_records import RECORDS_DIR, list_records

rows = list_records(limit=int(__import__("os").environ.get("BRACE_LIST_RECORDS_LIMIT", "20")))
if not rows:
    print("No records/ yet. Records are emitted automatically after each stage completes.")
    raise SystemExit(0)
for row in reversed(rows):
    print(f"{row['created_at']}  {row['stage']:24}  passed={row.get('passed')}  {row['path']}")
print(f"\nIndex: {RECORDS_DIR / 'index.jsonl'}")
PY
    ;;

  collect-trace-smoke)
    export BRACE_TRACED_ROLLOUT_DIR="${traced_rollout_dir}"
    export BRACE_TRACE_MAX_TRAJECTORIES="${BRACE_TRACE_MAX_TRAJECTORIES:-4}"
    export BRACE_ROLLOUT_WORKERS_PER_GPU="${BRACE_ROLLOUT_WORKERS_PER_GPU:-3}"
    export BRACE_GPU_IDS="${BRACE_GPU_IDS:-0 1 2 3 4 5 6 7}"
    exec bash experiments/brace/collect_traced_parallel.sh "$@"
    ;;

  collect-trace-audit)
    export BRACE_TRACED_ROLLOUT_DIR="${traced_rollout_dir}"
    unset BRACE_TRACE_MAX_TRAJECTORIES || true
    export BRACE_ROLLOUT_WORKERS_PER_GPU="${BRACE_ROLLOUT_WORKERS_PER_GPU:-3}"
    export BRACE_GPU_IDS="${BRACE_GPU_IDS:-0 1 2 3 4 5 6 7}"
    exec bash experiments/brace/collect_traced_parallel.sh "$@"
    ;;

  collect-trace-pilot)
    export BRACE_TRACED_ROLLOUT_DIR="${pilot_rollout_dir}"
    for task in "${tasks[@]}"; do
      seeds_file="$(pilot_seeds_file_for_task "${task}")"
      if [[ ! -s "${seeds_file}" ]]; then
        echo "Missing ${seeds_file}. Run select-pilot-seeds or select-confirm-seeds first." >&2
        exit 2
      fi
    done
    export BRACE_ROLLOUT_WORKERS_PER_GPU="${BRACE_ROLLOUT_WORKERS_PER_GPU:-3}"
    export BRACE_GPU_IDS="${BRACE_GPU_IDS:-0 1 2 3 4 5 6 7}"
    exec bash experiments/brace/collect_traced_parallel.sh "$@"
    ;;

  select-confirm-seeds)
    if [[ "${BRACE_LEGACY_MUTABLE_OUTPUTS:-0}" == "1" ]]; then
      seeds_output_dir="${brace_dir}/seeds"
    else
      seeds_output_dir="$(brace_stage_output_dir select_confirm_seeds seeds)"
    fi
    mkdir -p "${seeds_output_dir}"
    for task in "${tasks[@]}"; do
      exclude_file="${BRACE_EXCLUDE_SEEDS_FILE:-${brace_dir}/archive/branches_place_pilot_valid_v2.3/analyzed_seeds.json}"
      seeds_name="${task}_confirm_seeds.json"
      confirm_args=(
        --task "${task}"
        --rollout-dir "${traced_rollout_dir}"
        --count "${BRACE_CONFIRM_SEED_COUNT:-5}"
        --seed "${BRACE_CONFIRM_SEED_SELECTION:-1}"
        --exclude-seeds-file "${exclude_file}"
        --output "${seeds_output_dir}/${seeds_name}"
      )
      if [[ -n "${BRACE_CONFIRM_ROLLOUT_ID_MIN:-}" ]]; then
        confirm_args+=(--rollout-id-min "${BRACE_CONFIRM_ROLLOUT_ID_MIN}")
      fi
      if [[ -n "${BRACE_CONFIRM_ROLLOUT_ID_MAX:-}" ]]; then
        confirm_args+=(--rollout-id-max "${BRACE_CONFIRM_ROLLOUT_ID_MAX}")
      fi
      python experiments/brace/select_confirm_seeds.py "${confirm_args[@]}"
      if [[ "${BRACE_LEGACY_MUTABLE_OUTPUTS:-0}" != "1" ]]; then
        echo "${seeds_output_dir}" > "${brace_dir}/runs/LATEST_seeds_${seeds_name}"
      fi
    done
    ;;

  inventory-traced)
    mkdir -p "${brace_dir}/inventory"
    for task in "${tasks[@]}"; do
      python experiments/brace/inventory_traced_rollouts.py \
        --task "${task}" \
        --rollout-dir "${traced_rollout_dir}" \
        --output "${brace_dir}/inventory/${task}_traced.json"
    done
    ;;

  artifact-inventory)
    mkdir -p "${brace_dir}/sync/inventories"
    python experiments/brace/inventory_artifacts.py \
      --remote-repo "${BRACE_REMOTE_REPO:-/workspace/RoboTwin}" \
      "$@"
    ;;

  validate-artifacts)
    inventory_arg=()
    if [[ -n "${BRACE_SYNC_INVENTORY:-}" ]]; then
      inventory_arg=(--inventory "${BRACE_SYNC_INVENTORY}")
    fi
    python experiments/brace/validate_artifacts.py \
      "${inventory_arg[@]}" \
      --run-label "${dataset_run_label}" \
      "$@"
    ;;

  merge-audit-v2)
    merge_inputs=()
    if [[ -n "${BRACE_AUDIT_MERGE_INPUTS:-}" ]]; then
      read -r -a merge_inputs <<< "${BRACE_AUDIT_MERGE_INPUTS}"
    else
      for task in "${tasks[@]}"; do
        if summary_path="$(brace_latest_audit_summary "${task}")"; then
          merge_inputs+=("${summary_path}")
        else
          echo "No audit summary found for ${task}" >&2
          exit 2
        fi
      done
    fi
    if [[ "${BRACE_LEGACY_MUTABLE_OUTPUTS:-0}" == "1" ]]; then
      merge_output="${brace_dir}/replay_audit_v2/combined_summary.json"
    else
      merge_output_dir="$(brace_stage_output_dir merged_audit combined_audit)"
      merge_output="${merge_output_dir}/combined_summary.json"
    fi
    python experiments/brace/merge_replay_audit_summaries.py \
      --output "${merge_output}" \
      $(printf ' --input %q' "${merge_inputs[@]}")
    ;;

  promote-run)
    target="${BRACE_PROMOTE_TARGET:-}"
    if [[ -z "${target}" ]]; then
      echo "Set BRACE_PROMOTE_TARGET (e.g. archive/branches_place_pilot_valid_v2.3)" >&2
      exit 2
    fi
    promote_args=(--target "${target}")
    if [[ -n "${BRACE_PROMOTE_RUN:-}" ]]; then
      promote_args+=(--run-dir "${BRACE_PROMOTE_RUN}")
    fi
    if [[ -n "${dataset_run_label}" ]]; then
      promote_args+=(--run-label "${dataset_run_label}")
    fi
    if [[ "${BRACE_ALLOW_FAILED_GATE:-0}" == "1" ]]; then
      promote_args+=(--allow-failed-gate)
    fi
    python experiments/brace/promote_run.py "${promote_args[@]}" "$@"
    ;;

  audit-mutable-paths)
    python experiments/brace/audit_mutable_paths.py "$@"
    ;;

  archive-replay-gate)
    for task in "${tasks[@]}"; do
      audit_root="${BRACE_AUDIT_ROOT:-}"
      if [[ -z "${audit_root}" && -f "${brace_dir}/runs/LATEST_AUDIT_${task}" ]]; then
        audit_root="$(cat "${brace_dir}/runs/LATEST_AUDIT_${task}")"
      fi
      if [[ -z "${audit_root}" ]]; then
        audit_root="${brace_dir}/replay_audit_v2"
      fi
      echo "Archiving replay gate for ${task} from ${audit_root}" >&2
      python experiments/brace/extract_replay_gate_archive.py \
        --task "${task}" \
        --audit-root "${audit_root}"
    done
    ;;

  export-verified-chunks)
    if (( ${#tasks[@]} != 1 )); then
      echo "export-verified-chunks requires exactly one BRACE_TASKS task and a task-specific run label." >&2
      exit 2
    fi
    if [[ "${BRACE_LEGACY_MUTABLE_OUTPUTS:-0}" == "1" ]]; then
      export_output_dir="${dataset_dir}"
    else
      export_output_dir="$(brace_stage_output_dir export_verified_chunks "export_${dataset_run_label}")"
    fi
    mkdir -p "${export_output_dir}"
    if ! branch_source_dir="$(brace_latest_branch_dir "${BRACE_BRANCH_LABEL:-branches}")"; then
      echo "No branch dir found for export (set BRACE_BRANCH_LABEL or run branch first)" >&2
      exit 2
    fi
    traced_source_dir=${BRACE_TRACED_ROLLOUT_DIR:-${pilot_rollout_dir}}
    for task in "${tasks[@]}"; do
      python experiments/brace/export_verified_chunks.py \
        --protocol "${export_protocol}" \
        --branch-dir "${branch_source_dir}" \
        --rollout-dir "${traced_source_dir}" \
        --task "${task}" \
        --run-label "${dataset_run_label}" \
        --output-dir "${export_output_dir}" \
        --n1-seed "${BRACE_N1_SEED:-0}" \
        --workers "${BRACE_EXPORT_WORKERS:-96}"
    done
    if [[ "${BRACE_LEGACY_MUTABLE_OUTPUTS:-0}" != "1" ]]; then
      echo "Export run complete: ${export_output_dir}" >&2
      if [[ "${BRACE_AUTO_PROMOTE:-1}" == "1" ]]; then
        BRACE_PROMOTE_RUN="${export_output_dir}" \
        BRACE_PROMOTE_TARGET=datasets/ \
          bash experiments/brace/run_all.sh promote-run
      else
        echo "Promote to datasets/: BRACE_PROMOTE_RUN=${export_output_dir} BRACE_PROMOTE_TARGET=datasets/ bash experiments/brace/run_all.sh promote-run" >&2
      fi
    fi
    ;;

  prepare-hard-seeds)
    eval_dir="${BRACE_HARD_EVAL_DIR:-experiments/phase1/eval_results_200}"
    probe_dir="${eval_dir}/base_probe"
    hard_dir="${eval_dir}/hard_eval_seeds"
    hard_shards=${BRACE_HARD_PROBE_SHARDS:-8}
    mkdir -p "${probe_dir}" "${hard_dir}" "${brace_dir}/logs"
    if [[ "${BRACE_LEGACY_MUTABLE_OUTPUTS:-0}" == "1" ]]; then
      stage_run_dir="${brace_dir}/prepare_hard_seeds"
    else
      stage_run_dir="$(brace_stage_output_dir prepare_hard_seeds prepare_hard_seeds)"
    fi
    mkdir -p "${stage_run_dir}"
    stage_failed=0
    for task in "${tasks[@]}"; do
      ckpt_path="policy/DP/checkpoints/${task}-demo_clean-200-0/600.ckpt"
      hard_output="${hard_dir}/${task}.json"
      task_summary="${stage_run_dir}/${task}_summary.json"
      if [[ -s "${hard_output}" && "${BRACE_FORCE_HARD_SEEDS:-0}" != "1" ]]; then
        echo "Reusing existing hard seeds: ${hard_output}" >&2
        continue
      fi
      if ! run_hard_probe_sharded "${task}" "${ckpt_path}" "${probe_dir}"; then
        stage_failed=1
        continue
      fi
      if ! python experiments/brace/prepare_hard_seeds.py \
        --task "${task}" \
        --ckpt-path "${ckpt_path}" \
        --probe-dir "${probe_dir}" \
        --hard-output "${hard_output}" \
        --summary-output "${task_summary}" \
        --num-shards "${hard_shards}" \
        --merge-only; then
        stage_failed=1
      fi
    done
    if (( stage_failed != 0 )); then
      exit 1
    fi
    python - <<'PY' "${stage_run_dir}" "${tasks[@]}"
import json
import sys
from pathlib import Path

from experiments.brace.replay_audit import write_json_atomic
from experiments.brace.stage_records import emit_stage_record

stage_run_dir = Path(sys.argv[1])
tasks = sys.argv[2:]
summaries = []
for task in tasks:
    path = stage_run_dir / f"{task}_summary.json"
    if path.is_file():
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
aggregate = {
    "schema_version": 1,
    "stage": "prepare_hard_seeds",
    "passed": bool(summaries) and all(item.get("passed") for item in summaries),
    "tasks": {item["task"]: item for item in summaries},
}
write_json_atomic(stage_run_dir / "summary.json", aggregate)
emit_stage_record("prepare_hard_seeds", summary=aggregate, summary_path=stage_run_dir / "summary.json", tasks=tasks)
PY
    ;;

  anchor-smoke|anchor-feasibility|anchor-calibration|anchor-behavior-eval)
    echo "Stage '${stage}' is archived (BRACE-RW supersedes the single-timestep anchor route)." >&2
    echo "Frozen scripts: experiments/brace_v2_legacy/ (see its README.md)." >&2
    echo "Existing runs/ pointers and archive/ evidence are untouched." >&2
    exit 2
    ;;

  select-preservation-cohort)
    cohort_output_dir="$(brace_stage_output_dir select_preservation_cohort preservation_cohorts)"
    mkdir -p "${cohort_output_dir}"
    cohort_output="${cohort_output_dir}/${tasks[0]}_preservation_cohort.json"
    if [[ -z "${BRACE_CALIBRATION_RUN_DIR:-}" ]]; then
      BRACE_CALIBRATION_RUN_DIR="$(cat "${brace_dir}/runs/LATEST_anchor_calibration")"
    fi
    if [[ -z "${BRACE_CENSUS_DIR:-}" ]]; then
      if [[ -f "${brace_dir}/runs/LATEST_confirmatory_base_census" ]]; then
        BRACE_CENSUS_DIR="$(cat "${brace_dir}/runs/LATEST_confirmatory_base_census")"
      else
        echo "Run confirmatory-base-census first or set BRACE_CENSUS_DIR." >&2
        exit 2
      fi
    fi
    if [[ -z "${BRACE_BEHAVIOR_EVAL_DIR:-}" ]]; then
      if [[ -f "${brace_dir}/runs/LATEST_anchor_behavior_eval" ]]; then
        BRACE_BEHAVIOR_EVAL_DIR="$(cat "${brace_dir}/runs/LATEST_anchor_behavior_eval")"
      else
        echo "Set BRACE_BEHAVIOR_EVAL_DIR to the sealed Phase 3C behavior eval run." >&2
        exit 2
      fi
    fi
    python experiments/brace/select_preservation_cohort.py \
      --task "${tasks[0]}" \
      --census-dir "${BRACE_CENSUS_DIR}" \
      --calibration-run-dir "${BRACE_CALIBRATION_RUN_DIR}" \
      --behavior-eval-dir "${BRACE_BEHAVIOR_EVAL_DIR}" \
      --seeds-file "${confirmatory_seeds_file}" \
      --pilot-seeds-file "${brace_dir}/seeds/${tasks[0]}_pilot_seeds.json" \
      --confirm-seeds-file "${brace_dir}/seeds/${tasks[0]}_confirm_seeds.json" \
      --dataset-manifest "${BRACE_PRESERVATION_DATASET_MANIFEST:-${brace_dir}/datasets/${dataset_run_label}_N1.jsonl}" \
      --protocol "${confirmatory_protocol}" \
      --min-untouched "${BRACE_MIN_UNTOUCHED_PRESERVATION:-60}" \
      --output "${cohort_output}"
    echo "${cohort_output}" > "${brace_dir}/runs/LATEST_preservation_cohort"
    ;;

  confirmatory-base-census)
    census_run_dir="$(brace_stage_output_dir confirmatory_base_census confirmatory_base_census)"
    python experiments/brace/confirmatory_base_census.py \
      --task "${tasks[0]}" \
      --output "${census_run_dir}" \
      --protocol "${confirmatory_protocol}" \
      --seeds-file "${confirmatory_seeds_file}" \
      --workers-per-gpu "${EVAL_WORKERS_PER_GPU:-3}" \
      --gpu "${BRACE_CENSUS_GPU:-0}"
    echo "${census_run_dir}" > "${brace_dir}/runs/LATEST_confirmatory_base_census"
    ;;

  confirmatory-preservation)
    confirm_run_dir="$(brace_stage_output_dir confirmatory_preservation confirmatory_preservation)"
    if [[ ! -f "${brace_dir}/runs/LATEST_preservation_cohort" ]]; then
      echo "Run select-preservation-cohort first." >&2
      exit 2
    fi
    cohort_path="$(cat "${brace_dir}/runs/LATEST_preservation_cohort")"
    if [[ ! -f "${cohort_path}" ]] || [[ "$(jq -r '.meets_min_untouched // false' "${cohort_path}")" != "true" ]]; then
      echo "Blocked: preservation cohort is missing or does not meet the frozen untouched minimum: ${cohort_path}" >&2
      exit 2
    fi
    confirm_protocol_sha="$(sha256sum "${confirmatory_protocol}" | awk '{print $1}')"
    if [[ "$(jq -r '.protocol_sha256 // ""' "${cohort_path}")" != "${confirm_protocol_sha}" ]]; then
      echo "Blocked: cohort protocol SHA does not match ${confirmatory_protocol}." >&2
      exit 2
    fi
    # Publish the immutable run directory before launching so a failed
    # orchestrator remains directly discoverable and resumable.
    echo "${confirm_run_dir}" > "${brace_dir}/runs/LATEST_confirmatory_preservation"
    python experiments/brace/orchestrate_calibration.py \
      --protocol "${confirmatory_protocol}" \
      --jobs "${confirmatory_jobs}" \
      --task "${tasks[0]}" \
      --run-label "${dataset_run_label}" \
      --traced-rollout-dir "${BRACE_TRACED_ROLLOUT_DIR:-${traced_rollout_dir}}" \
      --run-dir "${confirm_run_dir}" \
      --gpus ${BRACE_GPU_IDS:-0 1 2 3 4 5 6 7} \
      --max-retries "${BRACE_CONFIRMATORY_MAX_RETRIES:-1}"
    ;;

  confirmatory-preservation-eval)
    eval_run_dir="$(brace_stage_output_dir confirmatory_preservation_eval confirmatory_preservation_eval)"
    python experiments/brace/orchestrate_confirmatory_eval.py \
      --task "${tasks[0]}" \
      --output "${eval_run_dir}" \
      --protocol "${confirmatory_protocol}" \
      --seeds-file "${confirmatory_seeds_file}" \
      --workers-per-gpu "${EVAL_WORKERS_PER_GPU:-3}" \
      --gpus ${BRACE_CONFIRMATORY_EVAL_GPU_IDS:-${BRACE_GPU_IDS:-0 1 2 3 4 5 6 7}} \
      --max-retries "${BRACE_CONFIRMATORY_EVAL_MAX_RETRIES:-1}" \
      --resume
    echo "${eval_run_dir}" > "${brace_dir}/runs/LATEST_confirmatory_preservation_eval"
    ;;

  confirmatory-preservation-report)
    eval_run_dir="${BRACE_CONFIRMATORY_EVAL_DIR:-$(cat "${brace_dir}/runs/LATEST_confirmatory_preservation_eval")}"
    cohort_path="$(cat "${brace_dir}/runs/LATEST_preservation_cohort")"
    census_dir="${BRACE_CENSUS_DIR:-$(cat "${brace_dir}/runs/LATEST_confirmatory_base_census")}"
    python experiments/brace/aggregate_confirmatory_preservation.py \
      --protocol "${confirmatory_protocol}" \
      --cohort "${cohort_path}" \
      --eval-dir "${eval_run_dir}" \
      --census-eval "${census_dir}/${tasks[0]}/census_base.json" \
      --output "${eval_run_dir}/h1_summary.json"
    ;;

  select-pilot-seeds)
    if [[ "${BRACE_LEGACY_MUTABLE_OUTPUTS:-0}" == "1" ]]; then
      seeds_output_dir="${brace_dir}/seeds"
    else
      seeds_output_dir="$(brace_stage_output_dir select_pilot_seeds seeds)"
    fi
    mkdir -p "${seeds_output_dir}"
    for task in "${tasks[@]}"; do
      seeds_name="${task}_pilot_seeds.json"
      python experiments/brace/select_pilot_seeds.py \
        --task "${task}" \
        --rollout-dir "${traced_rollout_dir}" \
        --count "${BRACE_PILOT_SEED_COUNT:-10}" \
        --seed "${BRACE_PILOT_SEED_SELECTION:-0}" \
        --output "${seeds_output_dir}/${seeds_name}"
      if [[ "${BRACE_LEGACY_MUTABLE_OUTPUTS:-0}" != "1" ]]; then
        echo "${seeds_output_dir}" > "${brace_dir}/runs/LATEST_seeds_${seeds_name}"
      fi
    done
    ;;

  verify-traced)
    if pgrep -af "experiments/brace/collect_traced_rollouts.py" >/dev/null 2>&1; then
      echo "Traced collection is still running." >&2
      exit 2
    fi
    for task in "${tasks[@]}"; do
      verify_rollout_dir="${BRACE_TRACED_ROLLOUT_DIR:-${traced_rollout_dir}}"
      verify_seeds_file="experiments/phase1/seeds/${task}_seeds.json"
      verify_shards="${num_shards}"
      if [[ "${verify_rollout_dir}" == *rollouts_traced_pilot* ]]; then
        verify_seeds_file="$(pilot_seeds_file_for_task "${task}")"
        verify_shards="$(pilot_shard_count_for_task "${task}")"
      fi
      python experiments/brace/verify_traced_rollouts.py \
        --tasks "${task}" \
        --rollout-dir "${verify_rollout_dir}" \
        --seeds-file "${verify_seeds_file}" \
        --rollouts-per-seed "${rollouts_per_seed}" \
        --num-shards "${verify_shards}" \
        --require-failures \
        --workers "${verify_workers}"

      python experiments/phase1/merge_rollout_shards.py \
        --task "${task}" \
        --task-config demo_brace_trace \
        --seeds-file "${verify_seeds_file}" \
        --rollouts-per-seed "${rollouts_per_seed}" \
        --rollout-dir "${verify_rollout_dir}"

      python experiments/brace/stage_records.py verify_traced \
        --artifact-dir "${verify_rollout_dir}/${task}" \
        --tasks "${task}" \
        --label traced_rollouts
    done
    echo "Traced rollouts verified."
    ;;

  branch)
    freeze_guard
    require_task_replay_gates
    if [[ ! -f experiments/brace/collect_branches.py ]]; then
      echo "collect_branches.py is not implemented yet; branch collection cannot start." >&2
      exit 2
    fi
    branch_label="${BRACE_BRANCH_LABEL:-branches}"
    if [[ -n "${BRACE_BRANCH_OUTPUT_DIR:-}" ]]; then
      branch_label="$(basename "${BRACE_BRANCH_OUTPUT_DIR}")"
    fi
    if [[ "${BRACE_LEGACY_MUTABLE_OUTPUTS:-0}" == "1" ]]; then
      branch_output_dir="${BRACE_BRANCH_OUTPUT_DIR:-${brace_dir}/${branch_label}}"
    else
      branch_output_dir="$(brace_branch_output_dir "${branch_label}")"
    fi
    echo "Immutable branch output: ${branch_output_dir}" >&2
    branch_rollout_dir=${BRACE_TRACED_ROLLOUT_DIR:-${pilot_rollout_dir}}
    branch_args=(
      --protocol "${protocol_v2}"
      --rollout-dir "${branch_rollout_dir}"
      --output-dir "${branch_output_dir}"
      --tasks "${tasks[@]}"
      --workers "${audit_workers}"
      --workers-per-gpu "${audit_workers_per_gpu}"
      --prepare-workers "${branch_prepare_workers}"
      --gpus "${gpu_ids[@]}"
    )
    if [[ -n "${BRACE_PILOT_SEEDS_FILE:-}" ]]; then
      branch_args+=(--seeds-file "${BRACE_PILOT_SEEDS_FILE}")
    fi
    exec python experiments/brace/collect_branches.py "${branch_args[@]}" "$@"
    ;;

  screen)
    freeze_guard
    block_dump_screen_until_place_pass
    require_task_replay_gates
    if ! branch_summary="$(brace_latest_branch_summary branches)"; then
      echo "Branch-quality gate has not passed: no branch summary found" >&2
      exit 2
    fi
    require_gate "${branch_summary}" "Branch-quality gate has not passed"
    if [[ ! -f experiments/brace/orchestrate.py ]]; then
      echo "BRACE orchestrate.py is not implemented yet; screen cannot start." >&2
      exit 2
    fi
    exec python experiments/brace/orchestrate.py screen \
      --protocol "${screen_protocol}" \
      --traced-rollout-dir "${BRACE_TRACED_ROLLOUT_DIR:-${traced_rollout_dir}}" \
      --gpus "${gpu_ids[@]}" \
      "$@"
    ;;

  full)
    freeze_guard
    require_v12_developmental_pass
    require_gate "${brace_dir}/promotions/screen.json" \
      "No preregistered BRACE screen promotion"
    if [[ ! -f experiments/brace/orchestrate.py ]]; then
      echo "BRACE orchestrate.py is not implemented yet; full evaluation cannot start." >&2
      exit 2
    fi
    exec python experiments/brace/orchestrate.py full \
      --protocol "${protocol}" \
      --gpus "${gpu_ids[@]}" \
      "$@"
    ;;

  *)
    echo "Unknown stage: ${stage}" >&2
    usage >&2
    exit 2
    ;;
esac
