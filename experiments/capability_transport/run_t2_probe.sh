#!/usr/bin/env bash
# T2 expert-source feasibility probe: scripted motion-planner expert on all
# 50 hard seeds of place_container_plate.
# Protocol: experiments/capability_transport/protocol.t2_expert_audit.v1.json
#
# Prerequisite (frozen ordering): the capability cells file
#   capability_cells.<task>.v1.json (+ .sha256)
# must exist BEFORE this launcher runs — the probe refuses otherwise.
#
# No policy checkpoint: the expert is the scripted planner (need_plan=True).
# 50 seeds are partitioned round-robin into lanes (default 2 lanes per GPU);
# each lane is one probe process working through its seed list serially and
# writing per-seed manifest shards. Re-running the same command resumes.
#
# Usage (inside the cloud container, from /workspace/RoboTwin):
#   bash experiments/capability_transport/run_t2_probe.sh place_container_plate
set -euo pipefail

TASK="${1:?usage: run_t2_probe.sh <task> [run_dir]}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CT_DIR="${REPO_ROOT}/experiments/capability_transport"

GPU_IDS=(${T2_GPU_IDS:-0 1 2 3 4 5 6 7})
LANES_PER_GPU="${T2_LANES_PER_GPU:-2}"
PYTHON_BIN="${T2_PYTHON_BIN:-python}"

CELLS_FILE="${CT_DIR}/capability_cells.${TASK}.v1.json"
GROUPS_FILE="${CT_DIR}/difficulty_groups.${TASK}.v1.json"

[[ -f "${CELLS_FILE}" && -f "${CELLS_FILE}.sha256" ]] || {
  echo "frozen cells file missing: ${CELLS_FILE} (+.sha256)." >&2
  echo "Run make_capability_cells.py FIRST — cells must be frozen before the probe." >&2
  exit 1
}
[[ -f "${GROUPS_FILE}" ]] || { echo "missing groups: ${GROUPS_FILE}" >&2; exit 1; }

if [[ $# -ge 2 ]]; then
  RUN_DIR="$2"
else
  LATEST="${CT_DIR}/runs/LATEST_T2_PROBE_${TASK}"
  if [[ -f "${LATEST}" ]] && [[ -d "$(cat "${LATEST}")" ]]; then
    RUN_DIR="$(cat "${LATEST}")"
  else
    RUN_DIR="${CT_DIR}/runs/$(date -u +%Y%m%dT%H%M%SZ)_t2_probe_${TASK}"
  fi
fi
mkdir -p "${RUN_DIR}"
echo "${RUN_DIR}" > "${CT_DIR}/runs/LATEST_T2_PROBE_${TASK}"

mapfile -t HARD_SEEDS < <("${PYTHON_BIN}" - "${GROUPS_FILE}" <<'EOF'
import json, sys
payload = json.load(open(sys.argv[1]))
hard = sorted(
    int(seed)
    for seed, info in payload["seeds"].items()
    if info["group"] == "hard"
)
print("\n".join(str(seed) for seed in hard))
EOF
)

N_LANES=$(( ${#GPU_IDS[@]} * LANES_PER_GPU ))
echo "Run dir: ${RUN_DIR}"
echo "Hard seeds: ${#HARD_SEEDS[@]}  lanes: ${N_LANES}  GPUs: ${GPU_IDS[*]}"

# Partition seeds round-robin into lanes.
declare -a LANE_SEEDS
for i in "${!HARD_SEEDS[@]}"; do
  lane=$(( i % N_LANES ))
  if [[ -z "${LANE_SEEDS[$lane]:-}" ]]; then
    LANE_SEEDS[$lane]="${HARD_SEEDS[$i]}"
  else
    LANE_SEEDS[$lane]="${LANE_SEEDS[$lane]},${HARD_SEEDS[$i]}"
  fi
done

PIDS=()
for lane in $(seq 0 $(( N_LANES - 1 ))); do
  seeds="${LANE_SEEDS[$lane]:-}"
  [[ -n "${seeds}" ]] || continue
  gpu="${GPU_IDS[$(( lane % ${#GPU_IDS[@]} ))]}"
  log="${RUN_DIR}/lane_${lane}.log"
  resume=()
  IFS=',' read -ra lane_arr <<< "${seeds}"
  for s in "${lane_arr[@]}"; do
    [[ -f "${RUN_DIR}/manifest_X.seed_${s}.jsonl" ]] && { resume=(--resume); break; }
  done
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" "${CT_DIR}/probe_expert_feasibility.py" \
      --task "${TASK}" --task-config demo_clean \
      --run-dir "${RUN_DIR}" --only-seeds "${seeds}" "${resume[@]}" \
      >> "${log}" 2>&1 &
  PIDS+=($!)
  echo "  GPU ${gpu} lane ${lane}: seeds ${seeds}  (log: ${log})"
done

echo "Launched ${#PIDS[@]} lanes"

FAIL=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || FAIL=1
done
if [[ "${FAIL}" -ne 0 ]]; then
  echo "One or more lanes failed; re-run the same command to resume." >&2
  exit 1
fi

"${PYTHON_BIN}" "${CT_DIR}/probe_expert_feasibility.py" \
  --task "${TASK}" --task-config demo_clean --run-dir "${RUN_DIR}" \
  --merge-shards

"${PYTHON_BIN}" "${CT_DIR}/probe_expert_feasibility.py" \
  --task "${TASK}" --task-config demo_clean --run-dir "${RUN_DIR}" \
  --report
echo "T2 probe finished: expert_source_feasibility.${TASK}.v1.json"
