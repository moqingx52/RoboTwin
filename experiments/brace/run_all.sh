#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

stage=${1:-help}
shift || true

read -r -a tasks <<< "${BRACE_TASKS:-place_container_plate dump_bin_bigbin}"
read -r -a gpu_ids <<< "${BRACE_GPU_IDS:-0 1 2 3 4 5 6 7}"

rollout_dir=${BRACE_ROLLOUT_DIR:-experiments/phase1/rollouts_200}
brace_dir=${BRACE_OUTPUT_DIR:-experiments/brace}
protocol=${BRACE_PROTOCOL_PATH:-${brace_dir}/protocol.json}
num_shards=${BRACE_NUM_SHARDS:-12}
rollouts_per_seed=${BRACE_ROLLOUTS_PER_SEED:-8}

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
  audit    Run deterministic prefix-replay audit (requires replay_audit.py).
  branch   Collect matched-continuation branches (requires passed replay audit).
  screen   Run B1/B2/B3/N1 screen (requires branch and anchor smoke gates).
  full     Run preregistered Base/U1/U4/B1/B2/B3 full evaluation.

Important:
  The old experiments/phase3 prep/screen/full stages are not called by this entry.
  The currently running failure collection is useful BRACE input and should finish.
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
      "$@"
    ;;

  branch)
    freeze_guard
    require_gate "${brace_dir}/replay_audit/summary.json" \
      "Replay audit gate has not passed"
    if [[ ! -f experiments/brace/collect_branches.py ]]; then
      echo "collect_branches.py is not implemented yet; branch collection cannot start." >&2
      exit 2
    fi
    exec python experiments/brace/collect_branches.py \
      --protocol "${protocol}" \
      --rollout-dir "${rollout_dir}" \
      --output-dir "${brace_dir}/branches" \
      --gpus "${gpu_ids[@]}" \
      "$@"
    ;;

  screen)
    freeze_guard
    require_gate "${brace_dir}/replay_audit/summary.json" \
      "Replay audit gate has not passed"
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
