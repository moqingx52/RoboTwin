#!/usr/bin/env bash
# Track A2: dump Stage-2 branch pilot (requires A1 replay gate passed).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

export BRACE_PROTOCOL_V2_PATH=experiments/brace/protocol.v2.3.json
export BRACE_TASKS=dump_bin_bigbin
export BRACE_ROLLOUT_WORKERS_PER_GPU="${BRACE_ROLLOUT_WORKERS_PER_GPU:-1}"
export BRACE_PILOT_NUM_SHARDS="${BRACE_PILOT_NUM_SHARDS:-8}"
unset BRACE_PILOT_SEEDS_FILE || true

brace_dir=experiments/brace
read -r -a tasks <<< "${BRACE_TASKS}"
# shellcheck source=experiments/brace/run_paths.sh
source "${repo_root}/experiments/brace/run_paths.sh"

if ! replay_summary="$(brace_latest_audit_summary dump_bin_bigbin)"; then
  echo "dump replay gate not passed; run A1 audit-v2 first." >&2
  python experiments/brace/inventory_traced_rollouts.py \
    --task dump_bin_bigbin \
    --rollout-dir experiments/brace/rollouts_traced \
    --output experiments/brace/inventory/dump_bin_bigbin_traced.json
  exit 2
fi
replay_gate="$(jq -r '.tasks.dump_bin_bigbin.replay_gate_passed // false' "${replay_summary}")"
if [[ "${replay_gate}" != "true" ]]; then
  echo "dump replay gate not passed; run A1 audit-v2 first." >&2
  python experiments/brace/inventory_traced_rollouts.py \
    --task dump_bin_bigbin \
    --rollout-dir experiments/brace/rollouts_traced \
    --output experiments/brace/inventory/dump_bin_bigbin_traced.json
  exit 2
fi

python experiments/brace/inventory_traced_rollouts.py \
  --task dump_bin_bigbin \
  --rollout-dir experiments/brace/rollouts_traced \
  --output experiments/brace/inventory/dump_bin_bigbin_traced.json

mixed_count="$(jq '.mixed_outcome_count' experiments/brace/inventory/dump_bin_bigbin_traced.json)"
if [[ "${mixed_count}" -lt 10 ]]; then
  echo "Need >= 10 mixed-outcome seeds, found ${mixed_count}. Direct targeted collection before branch." >&2
  exit 2
fi

echo "=== A2 Step 1/4: select 10 mixed-outcome pilot seeds ==="
BRACE_TRACED_ROLLOUT_DIR=experiments/brace/rollouts_traced \
  bash experiments/brace/run_all.sh select-pilot-seeds

export BRACE_TRACED_ROLLOUT_DIR=experiments/brace/rollouts_traced_pilot
echo "=== A2 Step 2/4: collect traced pilot rollouts (10 seeds, capped shards) ==="
bash experiments/brace/run_all.sh collect-trace-pilot
echo "=== A2 Step 3/4: verify traced pilot rollouts ==="
bash experiments/brace/run_all.sh verify-traced

export BRACE_BRANCH_LABEL=branches_dump
echo "=== A2 Step 4/4: branch collection -> runs/.../branches_dump/ ==="
bash experiments/brace/run_all.sh branch

if branch_dir="$(brace_latest_branch_dir branches_dump)"; then
  echo "Dump branch pilot complete: ${branch_dir}/summary.json"
  bash experiments/brace/archive_dump_pilot.sh
else
  echo "Dump branch pilot finished; check runs/LATEST_branches_dump" >&2
fi
