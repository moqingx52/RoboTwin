#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

export BRACE_TASKS=place_container_plate
export BRACE_TRACED_ROLLOUT_DIR="${BRACE_TRACED_ROLLOUT_DIR:-experiments/brace/rollouts_traced_pilot}"

bash experiments/brace/run_all.sh select-pilot-seeds
bash experiments/brace/run_all.sh collect-trace-pilot
BRACE_TRACED_ROLLOUT_DIR="${BRACE_TRACED_ROLLOUT_DIR}" bash experiments/brace/run_all.sh verify-traced
BRACE_TASKS=place_container_plate BRACE_TRACED_ROLLOUT_DIR="${BRACE_TRACED_ROLLOUT_DIR}" bash experiments/brace/run_all.sh audit-v2
BRACE_TASKS=place_container_plate BRACE_TRACED_ROLLOUT_DIR="${BRACE_TRACED_ROLLOUT_DIR}" bash experiments/brace/run_all.sh branch
