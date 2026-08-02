#!/usr/bin/env bash
# Timestamped immutable run directories for BRACE stages.
#
# Policy: never overwrite shared mutable JSON under experiments/brace/replay_audit_v2/
# or experiments/brace/branches/ unless BRACE_LEGACY_MUTABLE_OUTPUTS=1.

brace_utc_run_id() {
  date -u +%Y%m%dT%H%M%SZ
}

brace_task_slug() {
  local joined
  joined="$(printf '%s' "${tasks[*]}" | tr ' ' '_')"
  printf '%s' "${joined}"
}

brace_allocate_run_dir() {
  local stage=$1
  local subpath=${2:-}
  if [[ "${BRACE_LEGACY_MUTABLE_OUTPUTS:-0}" == "1" ]]; then
    if [[ -n "${subpath}" ]]; then
      BRACE_ACTIVE_RUN_DIR="${brace_dir}/${subpath}"
    else
      BRACE_ACTIVE_RUN_DIR="${brace_dir}"
    fi
    mkdir -p "${BRACE_ACTIVE_RUN_DIR}"
    printf '%s\n' "${BRACE_ACTIVE_RUN_DIR}"
    return 0
  fi

  BRACE_RUN_ID="${BRACE_RUN_ID:-$(brace_utc_run_id)}"
  local run_name="${BRACE_RUN_ID}_${stage}"
  if ((${#tasks[@]} == 1)); then
    run_name="${run_name}_${tasks[0]}"
  elif ((${#tasks[@]} > 1)); then
    run_name="${run_name}_$(brace_task_slug)"
  fi
  BRACE_ACTIVE_RUN_DIR="${brace_dir}/runs/${run_name}"
  mkdir -p "${BRACE_ACTIVE_RUN_DIR}"

  local -a meta_args=(
    --run-dir "${BRACE_ACTIVE_RUN_DIR}"
    --stage "${stage}"
    --run-id "${BRACE_RUN_ID}"
  )
  local task
  for task in "${tasks[@]}"; do
    meta_args+=(--tasks "${task}")
  done
  if [[ -n "${subpath}" ]]; then
    meta_args+=(--extra "subpath=${subpath}")
  fi
  python experiments/brace/write_run_meta.py "${meta_args[@]}" >/dev/null

  mkdir -p "${brace_dir}/runs"
  printf '%s\n' "${BRACE_RUN_ID}" > "${brace_dir}/runs/LATEST_RUN_ID"
  printf '%s\n' "${BRACE_ACTIVE_RUN_DIR}" > "${brace_dir}/runs/LATEST"
  printf '%s\n' "${BRACE_ACTIVE_RUN_DIR}"
}

brace_audit_output_dir() {
  local task=$1
  if [[ "${BRACE_LEGACY_MUTABLE_OUTPUTS:-0}" == "1" ]]; then
    if [[ -n "${BRACE_AUDIT_OUTPUT_DIR:-}" ]]; then
      printf '%s\n' "${BRACE_AUDIT_OUTPUT_DIR}"
      return 0
    fi
    printf '%s\n' "${brace_dir}/replay_audit_v2"
    return 0
  fi

  if [[ -z "${BRACE_RUN_ROOT:-}" ]]; then
    BRACE_RUN_ROOT="$(brace_allocate_run_dir audit_v2 replay_audit_v2)"
    export BRACE_RUN_ROOT
  fi
  local out="${BRACE_RUN_ROOT}/replay_audit_v2/${task}"
  mkdir -p "${out}"
  echo "${out}" > "${brace_dir}/runs/LATEST_AUDIT_${task}"
  printf '%s\n' "${out}"
}

brace_branch_output_dir() {
  local label=${1:-branches}
  if [[ "${BRACE_LEGACY_MUTABLE_OUTPUTS:-0}" == "1" ]]; then
    printf '%s\n' "${BRACE_BRANCH_OUTPUT_DIR:-${brace_dir}/${label}}"
    return 0
  fi
  if [[ -z "${BRACE_RUN_ROOT:-}" ]]; then
    BRACE_RUN_ROOT="$(brace_allocate_run_dir branch "${label}")"
    export BRACE_RUN_ROOT
  fi
  local out="${BRACE_RUN_ROOT}/${label}"
  mkdir -p "${out}"
  echo "${out}" > "${brace_dir}/runs/LATEST_${label}"
  printf '%s\n' "${out}"
}

brace_audit_task_gate_passed() {
  local summary=$1
  local task=$2
  [[ "$(jq -r --arg t "${task}" '.tasks[$t].replay_gate_passed // false' "${summary}")" == "true" ]]
}

brace_audit_archive_summary() {
  local task=$1
  local archive_name="replay_audit_v2_place_v2.3_gate"
  if [[ "${task}" == "dump_bin_bigbin" ]]; then
    archive_name="replay_audit_v2_dump_v2.3_gate"
  fi
  local candidate="${brace_dir}/archive/${archive_name}/summary.json"
  if [[ -s "${candidate}" ]] && brace_audit_task_gate_passed "${candidate}" "${task}"; then
    printf '%s\n' "${candidate}"
    return 0
  fi
  return 1
}

brace_latest_audit_summary() {
  local task=$1
  local pointer candidate summary
  if [[ -n "${BRACE_AUDIT_RUN_DIR:-}" ]]; then
    summary="${BRACE_AUDIT_RUN_DIR}/summary.json"
    if [[ ! -s "${summary}" ]]; then
      echo "BRACE_AUDIT_RUN_DIR has no summary.json: ${BRACE_AUDIT_RUN_DIR}" >&2
      return 1
    fi
    if ! brace_audit_task_gate_passed "${summary}" "${task}"; then
      echo "BRACE_AUDIT_RUN_DIR replay gate not passed for ${task}: ${summary}" >&2
      return 1
    fi
    printf '%s\n' "${summary}"
    return 0
  fi
  pointer="$(cat "${brace_dir}/runs/LATEST_AUDIT_${task}" 2>/dev/null || true)"
  if [[ -n "${pointer}" && -s "${pointer}/summary.json" ]]; then
    summary="${pointer}/summary.json"
    if brace_audit_task_gate_passed "${summary}" "${task}"; then
      printf '%s\n' "${summary}"
      return 0
    fi
    echo "WARNING: latest audit failed replay gate, skipping immutable run: ${pointer}" >&2
  fi
  if summary="$(brace_audit_archive_summary "${task}")"; then
    printf '%s\n' "${summary}"
    return 0
  fi
  for candidate in \
    "${brace_dir}/replay_audit_v2/${task}/summary.json" \
    "${brace_dir}/replay_audit_v2/summary.json"; do
    if [[ -s "${candidate}" ]] && brace_audit_task_gate_passed "${candidate}" "${task}"; then
      echo "WARNING: using legacy audit summary (deprecated): ${candidate}" >&2
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

brace_latest_branch_dir() {
  local label=${1:-branches}
  local pointer candidate
  if [[ -n "${BRACE_BRANCH_DIR:-}" && -s "${BRACE_BRANCH_DIR}/summary.json" ]]; then
    printf '%s\n' "${BRACE_BRANCH_DIR}"
    return 0
  fi
  pointer="$(cat "${brace_dir}/runs/LATEST_${label}" 2>/dev/null || true)"
  if [[ -n "${pointer}" && -s "${pointer}/summary.json" ]]; then
    printf '%s\n' "${pointer}"
    return 0
  fi
  local archive_name="branches_place_pilot_valid_v2.3"
  case "${label}" in
    branches_confirm) archive_name="branches_place_confirm_v2.3" ;;
    branches_dump) archive_name="branches_dump_pilot_valid_v2.3" ;;
  esac
  for candidate in \
    "${brace_dir}/archive/${archive_name}" \
    "${brace_dir}/${label}"; do
    if [[ -s "${candidate}/summary.json" ]]; then
      if [[ "${candidate}" == "${brace_dir}/${label}" ]]; then
        echo "WARNING: using legacy branch dir (deprecated): ${candidate}" >&2
      fi
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

brace_latest_branch_summary() {
  local label=${1:-branches}
  local branch_dir
  if branch_dir="$(brace_latest_branch_dir "${label}")"; then
    printf '%s\n' "${branch_dir}/summary.json"
    return 0
  fi
  return 1
}

brace_stage_output_dir() {
  local stage=$1
  local pointer_suffix=${2:-}
  if [[ "${BRACE_LEGACY_MUTABLE_OUTPUTS:-0}" == "1" ]]; then
    if [[ -n "${BRACE_STAGE_OUTPUT_DIR:-}" ]]; then
      mkdir -p "${BRACE_STAGE_OUTPUT_DIR}"
      printf '%s\n' "${BRACE_STAGE_OUTPUT_DIR}"
      return 0
    fi
    printf '%s\n' "${brace_dir}"
    return 0
  fi
  if [[ -z "${BRACE_RUN_ROOT:-}" ]]; then
    BRACE_RUN_ROOT="$(brace_allocate_run_dir "${stage}" "${pointer_suffix}")"
    export BRACE_RUN_ROOT
  fi
  if [[ -n "${pointer_suffix}" ]]; then
    echo "${BRACE_RUN_ROOT}" > "${brace_dir}/runs/LATEST_${pointer_suffix}"
  fi
  printf '%s\n' "${BRACE_RUN_ROOT}"
}

brace_resolve_audit_summary() {
  local task=$1
  brace_latest_audit_summary "${task}"
}

brace_resolve_seeds_file() {
  local name=$1
  local pointer candidate
  pointer="$(cat "${brace_dir}/runs/LATEST_seeds_${name}" 2>/dev/null || true)"
  if [[ -n "${pointer}" ]]; then
    candidate="${pointer}/${name}"
    if [[ -s "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  fi
  candidate="${brace_dir}/seeds/${name}"
  if [[ -s "${candidate}" ]]; then
    printf '%s\n' "${candidate}"
    return 0
  fi
  return 1
}
