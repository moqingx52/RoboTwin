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
screen_protocol=${BRACE_SCREEN_PROTOCOL_PATH:-${brace_dir}/screen_protocol.v1.json}

pilot_seeds_file_for_task() {
  local task=$1
  if [[ -n "${BRACE_PILOT_SEEDS_FILE:-}" ]]; then
    echo "${BRACE_PILOT_SEEDS_FILE}"
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
  export-verified-chunks  Export B1/N1 chunk manifests from branch artifacts.
  anchor-smoke         Frozen-denoiser anchor structural smoke (screen_protocol.v1).
  branch   Collect matched-continuation branches (requires passed replay audit v2).
  screen   Run B1/B2/B3/N1 screen (requires branch and anchor smoke gates).
  full     Run preregistered Base/U1/U4/B1/B2/B3 full evaluation.

Important:
  The old experiments/phase3 prep/screen/full stages are not called by this entry.
  The currently running failure collection is useful BRACE input and should finish.

Environment (v2 audit / branch):
  BRACE_PROTOCOL_V2_PATH   Protocol file for audit-v2 and branch (default:
                           experiments/brace/protocol.v2.3.json). Set explicitly when
                           reproducing archived v2.3 runs.
  BRACE_TRACED_ROLLOUT_DIR Traced HDF5 root (e.g. rollouts_traced_pilot for Stage 2).
  BRACE_BRANCH_OUTPUT_DIR  Branch summary/checks output dir (default: experiments/brace/branches).
  BRACE_PILOT_SEEDS_FILE   Override pilot/confirm seeds JSON for collect/verify/branch.
  BRACE_TASKS              Space-separated task subset (default: both protocol tasks).
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
  local path=$1
  local task task_summary
  for task in "${tasks[@]}"; do
    task_summary="${brace_dir}/replay_audit_v2/${task}/summary.json"
    if [[ -s "${task_summary}" ]]; then
      if [[ "$(jq -r '.tasks[$task].replay_gate_passed // false' --arg task "${task}" "${task_summary}")" != "true" ]]; then
        echo "Replay audit v2 per-task gate has not passed for ${task}: ${task_summary}" >&2
        exit 2
      fi
      continue
    fi
    if [[ ! -s "${path}" ]] || [[ "$(jq -r --arg task "${task}" '.tasks[$task].replay_gate_passed // false' "${path}")" != "true" ]]; then
      echo "Replay audit v2 per-task gate has not passed for ${task}: ${path}" >&2
      exit 2
    fi
  done
}

case "${stage}" in
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
    audit_output_dir="${BRACE_AUDIT_OUTPUT_DIR:-${brace_dir}/replay_audit_v2}"
    exec python experiments/brace/replay_audit_v2.py \
      --protocol "${protocol_v2}" \
      --rollout-dir "${traced_rollout_dir}" \
      --output-dir "${audit_output_dir}" \
      --tasks "${tasks[@]}" \
      --workers "${audit_workers}" \
      --workers-per-gpu "${audit_workers_per_gpu}" \
      --gpus "${gpu_ids[@]}" \
      "$@"
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
    mkdir -p "${brace_dir}/seeds"
    for task in "${tasks[@]}"; do
      exclude_file="${BRACE_EXCLUDE_SEEDS_FILE:-${brace_dir}/archive/branches_place_pilot_valid_v2.3/analyzed_seeds.json}"
      confirm_args=(
        --task "${task}"
        --rollout-dir "${traced_rollout_dir}"
        --count "${BRACE_CONFIRM_SEED_COUNT:-5}"
        --seed "${BRACE_CONFIRM_SEED_SELECTION:-1}"
        --exclude-seeds-file "${exclude_file}"
        --output "${brace_dir}/seeds/${task}_confirm_seeds.json"
      )
      if [[ -n "${BRACE_CONFIRM_ROLLOUT_ID_MIN:-}" ]]; then
        confirm_args+=(--rollout-id-min "${BRACE_CONFIRM_ROLLOUT_ID_MIN}")
      fi
      if [[ -n "${BRACE_CONFIRM_ROLLOUT_ID_MAX:-}" ]]; then
        confirm_args+=(--rollout-id-max "${BRACE_CONFIRM_ROLLOUT_ID_MAX}")
      fi
      python experiments/brace/select_confirm_seeds.py "${confirm_args[@]}"
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
        merge_inputs+=("${brace_dir}/replay_audit_v2/${task}/summary.json")
      done
    fi
    python experiments/brace/merge_replay_audit_summaries.py \
      --output "${brace_dir}/replay_audit_v2/combined_summary.json" \
      $(printf ' --input %q' "${merge_inputs[@]}")
    ;;

  archive-replay-gate)
    for task in "${tasks[@]}"; do
      BRACE_ARCHIVE_TASK="${task}" \
      BRACE_AUDIT_ROOT="${BRACE_AUDIT_ROOT:-${brace_dir}/replay_audit_v2}" \
        bash experiments/brace/archive_replay_gate.sh
    done
    ;;

  export-verified-chunks)
    mkdir -p "${dataset_dir}"
    branch_source_dir=${BRACE_BRANCH_DIR:-${brace_dir}/branches}
    traced_source_dir=${BRACE_TRACED_ROLLOUT_DIR:-${pilot_rollout_dir}}
    for task in "${tasks[@]}"; do
      python experiments/brace/export_verified_chunks.py \
        --protocol "${protocol_v2}" \
        --branch-dir "${branch_source_dir}" \
        --rollout-dir "${traced_source_dir}" \
        --task "${task}" \
        --run-label "${dataset_run_label}" \
        --output-dir "${dataset_dir}" \
        --n1-seed "${BRACE_N1_SEED:-0}" \
        --workers "${BRACE_EXPORT_WORKERS:-96}"
    done
    ;;

  anchor-smoke)
    mkdir -p "${brace_dir}/anchor_smoke"
    python experiments/brace/anchor_smoke.py \
      --protocol "${screen_protocol}" \
      --output "${brace_dir}/anchor_smoke/summary.json"
    ;;

  select-pilot-seeds)
    mkdir -p "${brace_dir}/seeds"
    for task in "${tasks[@]}"; do
      python experiments/brace/select_pilot_seeds.py \
        --task "${task}" \
        --rollout-dir "${traced_rollout_dir}" \
        --count "${BRACE_PILOT_SEED_COUNT:-10}" \
        --seed "${BRACE_PILOT_SEED_SELECTION:-0}" \
        --output "${brace_dir}/seeds/${task}_pilot_seeds.json"
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
    done
    echo "Traced rollouts verified."
    ;;

  branch)
    freeze_guard
    require_task_replay_gates "${brace_dir}/replay_audit_v2/summary.json"
    if [[ ! -f experiments/brace/collect_branches.py ]]; then
      echo "collect_branches.py is not implemented yet; branch collection cannot start." >&2
      exit 2
    fi
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
    require_gate "${brace_dir}/replay_audit_v2/summary.json" \
      "Replay audit v2 gate has not passed"
    require_gate "${brace_dir}/branches/summary.json" \
      "Branch-quality gate has not passed"
    require_gate "${brace_dir}/anchor_smoke/summary.json" \
      "Frozen-denoiser anchor smoke gate has not passed"
    if [[ ! -f experiments/brace/orchestrate.py ]]; then
      echo "BRACE orchestrate.py is not implemented yet; screen cannot start." >&2
      exit 2
    fi
    exec python experiments/brace/orchestrate.py screen \
      --protocol "${protocol}" \
      --gpus "${gpu_ids[@]}" \
      "$@"
    ;;

  full)
    freeze_guard
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
