#!/usr/bin/env bash
# Track A2: dump Stage-2 branch pilot (requires A1 replay gate passed).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

export BRACE_PROTOCOL_V2_PATH=experiments/brace/protocol.v2.3.json
export BRACE_TASKS=dump_bin_bigbin
unset BRACE_PILOT_SEEDS_FILE || true

replay_gate="$(jq -r '.tasks.dump_bin_bigbin.replay_gate_passed // false' experiments/brace/replay_audit_v2/summary.json)"
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

BRACE_TRACED_ROLLOUT_DIR=experiments/brace/rollouts_traced \
  bash experiments/brace/run_all.sh select-pilot-seeds

export BRACE_TRACED_ROLLOUT_DIR=experiments/brace/rollouts_traced_pilot
bash experiments/brace/run_all.sh collect-trace-pilot
bash experiments/brace/run_all.sh verify-traced

export BRACE_BRANCH_OUTPUT_DIR=experiments/brace/branches_dump
bash experiments/brace/run_all.sh branch

echo "Dump branch pilot complete: experiments/brace/branches_dump/summary.json"
