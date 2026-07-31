#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

export BRACE_TASKS=place_container_plate
export BRACE_TRACED_ROLLOUT_DIR="${BRACE_TRACED_ROLLOUT_DIR:-experiments/brace/rollouts_traced_pilot}"
export BRACE_TRACE_ENV_SEEDS="${BRACE_TRACE_ENV_SEEDS:-$(jq -r '.seeds | join(" ")' experiments/brace/seeds/place_container_plate_pilot_seeds.json)}"

bash experiments/brace/run_all.sh select-pilot-seeds || true
bash experiments/brace/run_all.sh collect-trace-pilot
bash experiments/brace/run_all.sh verify-traced
BRACE_TASKS=place_container_plate bash experiments/brace/run_all.sh branch
