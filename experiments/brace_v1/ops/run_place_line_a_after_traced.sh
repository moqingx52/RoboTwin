#!/bin/bash
# Run after place Base200 traced corpus is verified/merged under
# experiments/brace/rollouts_traced_base200_v2. Does not start held-out BRACE.
#
# Stages: promote existing audit (or re-run) → branch → export verified/matched chunks.
# Do NOT call run_place_pilot.sh (that re-collects rollouts_traced_pilot).
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
source /root/miniconda/etc/profile.d/conda.sh
conda activate RoboTwin
export PYTHONPATH="${repo_root}:${repo_root}/policy/DP${PYTHONPATH:+:${PYTHONPATH}}"

task=${BRACE_TASKS:-place_container_plate}
traced=${BRACE_TRACED_ROLLOUT_DIR:-experiments/brace/rollouts_traced_base200_v2}
state=${BRACE_PLACE_TRACED_STATE:-experiments/brace/runs/base200_place_traced_15shard_20260811.state.json}
read -r -a gpu_ids <<< "${BRACE_GPU_IDS:-0 1 2 4 5 6 7}"

if [[ ! -f "${state}" ]]; then
  echo "missing traced state: ${state}" >&2
  exit 2
fi
python - "${state}" <<'PY'
import json, sys
s=json.load(open(sys.argv[1], encoding="utf-8"))
if not s.get("place_eval_gate_passed"):
    raise SystemExit("place eval gate not passed")
if not s.get("verify_done"):
    raise SystemExit("traced verify_done is false; refuse to start after-traced")
print("traced_gate_ok")
PY

export BRACE_TASKS="${task}"
export BRACE_TRACED_ROLLOUT_DIR="${traced}"
export BRACE_GPU_IDS="${gpu_ids[*]}"
export BRACE_AUDIT_WORKERS_PER_GPU="${BRACE_AUDIT_WORKERS_PER_GPU:-3}"
export BRACE_AUDIT_WORKERS=$(( ${#gpu_ids[@]} * BRACE_AUDIT_WORKERS_PER_GPU ))
export BRACE_AUDIT_PROMOTE_TARGET="${BRACE_AUDIT_PROMOTE_TARGET:-archive/replay_audit_v2_place_base200_v2_gate}"
export BRACE_BRANCH_LABEL="${BRACE_BRANCH_LABEL:-branches_place_base200_v2}"
export BRACE_DATASET_RUN_LABEL="${BRACE_DATASET_RUN_LABEL:-place_base200_v2}"

audit_ptr="experiments/brace/runs/LATEST_AUDIT_${task}"
if [[ ! -f "${audit_ptr}" ]]; then
  echo "=== stage: replay audit v2 ==="
  bash experiments/brace/run_all.sh audit-v2
else
  audit_root="$(cat "${audit_ptr}")"
  summary="${audit_root}/replay_audit_v2/${task}/summary.json"
  if [[ ! -f "${summary}" ]] || ! python - "${summary}" <<'PY'
import json, sys
s=json.load(open(sys.argv[1], encoding="utf-8"))
task="place_container_plate"
ok=bool(s.get("passed")) and bool((s.get("tasks") or {}).get(task, {}).get("replay_gate_passed"))
raise SystemExit(0 if ok else 2)
PY
  then
    echo "=== stage: replay audit v2 (re-run; previous summary missing/failed) ==="
    bash experiments/brace/run_all.sh audit-v2
  else
    echo "=== stage: promote passed audit to ${BRACE_AUDIT_PROMOTE_TARGET} ==="
    BRACE_PROMOTE_RUN="${audit_root}/replay_audit_v2/${task}" \
    BRACE_PROMOTE_TARGET="${BRACE_AUDIT_PROMOTE_TARGET}" \
      bash experiments/brace/run_all.sh promote-run
  fi
fi

echo "=== stage: branch (matched continuation + controls) ==="
bash experiments/brace/run_all.sh branch

echo "=== stage: export verified chunks / matched-random controls ==="
bash experiments/brace/run_all.sh export-verified-chunks

echo "place Line A after-traced: audit+branch+export complete."
echo "Next: launch place Base200 developmental U0/N1/B1/B2/B3 (do not start held-out BRACE)."
echo "Freeze BRACE-v2 only after preservation/adaptation gates pass."
