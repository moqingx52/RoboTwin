#!/usr/bin/env bash
# P0: freeze place confirmatory branch artifacts via promote-run.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

pilot_archive=experiments/brace/archive/branches_place_pilot_valid_v2.3
confirm_run="${BRACE_PROMOTE_RUN:-}"
if [[ -z "${confirm_run}" && -f experiments/brace/runs/LATEST_branches_confirm ]]; then
  confirm_run="$(cat experiments/brace/runs/LATEST_branches_confirm)"
fi
if [[ -z "${confirm_run}" || ! -s "${confirm_run}/summary.json" ]]; then
  echo "Set BRACE_PROMOTE_RUN or run confirm branch first (runs/LATEST_branches_confirm)." >&2
  exit 2
fi

if [[ ! -f "${pilot_archive}/summary.json" ]]; then
  echo "Missing pilot archive summary: ${pilot_archive}/summary.json" >&2
  exit 2
fi

merged_output="${confirm_run}/merged_gate.json"
python experiments/brace/evaluate_confirmatory_gate.py \
  --pilot-summary "${pilot_archive}/summary.json" \
  --confirm-summary "${confirm_run}/summary.json" \
  --output "${merged_output}" || true

BRACE_PROMOTE_RUN="${confirm_run}" \
BRACE_PROMOTE_TARGET=archive/branches_place_confirm_v2.3 \
BRACE_ALLOW_FAILED_GATE=1 \
  bash experiments/brace/run_all.sh promote-run

echo "Archived via promote-run to experiments/brace/archive/branches_place_confirm_v2.3"
