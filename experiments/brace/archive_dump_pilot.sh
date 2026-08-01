#!/usr/bin/env bash
# P0: freeze dump pilot branch artifacts via promote-run.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

branch_run="${BRACE_PROMOTE_RUN:-}"
if [[ -z "${branch_run}" && -f experiments/brace/runs/LATEST_branches_dump ]]; then
  branch_run="$(cat experiments/brace/runs/LATEST_branches_dump)"
fi
if [[ -z "${branch_run}" ]]; then
  echo "Set BRACE_PROMOTE_RUN or run dump branch first (runs/LATEST_branches_dump)." >&2
  exit 2
fi

BRACE_PROMOTE_RUN="${branch_run}" \
BRACE_PROMOTE_TARGET=archive/branches_dump_pilot_valid_v2.3 \
  bash experiments/brace/run_all.sh promote-run

echo "Archived via promote-run to experiments/brace/archive/branches_dump_pilot_valid_v2.3"
