#!/usr/bin/env bash
# T1c source acquisition: round-robin success-first collection with pi0.
# Protocol: experiments/capability_transport/protocol.t1.v1.2.json
# (D_E/D_M target mode, serial within group; D_H fixed opportunity budget,
#  one process per supported-hard seed — seeds are independent.)
#
# Packing on the measured 8x4090 / 96-core cloud:
#   15 D_H seed workers + 1 D_E + 1 D_M = 17 simulator processes, assigned
#   round-robin across T1_GPU_IDS (default 0-7). D_E/D_M stay serial because
#   their stop condition is evaluated before every attempt.
#
# Re-running the same command resumes in place (append-only manifests).
#
# Usage (inside the cloud container, from /workspace/RoboTwin):
#   bash experiments/capability_transport/run_t1c_collection.sh place_container_plate
set -euo pipefail

TASK="${1:?usage: run_t1c_collection.sh <task> [run_dir]}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CT_DIR="${REPO_ROOT}/experiments/capability_transport"

GPU_IDS=(${T1_GPU_IDS:-0 1 2 3 4 5 6 7})
PYTHON_BIN="${T1_PYTHON_BIN:-python}"

CKPT="${REPO_ROOT}/policy/DP/checkpoints/${TASK}-demo_clean-200-0/600.ckpt"
GROUPS_FILE="${CT_DIR}/difficulty_groups.${TASK}.v1.json"

# Refuse tasks that failed the source-feasibility gate.
"${PYTHON_BIN}" - "$TASK" "${CT_DIR}/t1c_source_feasibility.v1.json" <<'EOF'
import json, sys
task, path = sys.argv[1], sys.argv[2]
gate = json.load(open(path))["tasks"]
if task not in gate:
    sys.exit(f"{task} has no source-feasibility record in {path}")
verdict = gate[task]["verdict"]
if not verdict.startswith("PASS"):
    sys.exit(f"{task} failed the T1c source-feasibility gate: {verdict}")
print(f"Feasibility gate: {verdict}")
EOF

if [[ $# -ge 2 ]]; then
  RUN_DIR="$2"
else
  LATEST="${CT_DIR}/runs/LATEST_T1C_COLLECTION_${TASK}"
  if [[ -f "${LATEST}" ]] && [[ -d "$(cat "${LATEST}")" ]]; then
    RUN_DIR="$(cat "${LATEST}")"
  else
    RUN_DIR="${CT_DIR}/runs/$(date -u +%Y%m%dT%H%M%SZ)_t1c_collection_${TASK}"
  fi
fi
mkdir -p "${RUN_DIR}"
echo "${RUN_DIR}" > "${CT_DIR}/runs/LATEST_T1C_COLLECTION_${TASK}"

[[ -f "${CKPT}" ]] || { echo "missing checkpoint: ${CKPT}" >&2; exit 1; }
[[ -f "${GROUPS_FILE}" ]] || { echo "missing groups: ${GROUPS_FILE}" >&2; exit 1; }

mapfile -t DH_SEEDS < <("${PYTHON_BIN}" - "${GROUPS_FILE}" <<'EOF'
import json, sys
payload = json.load(open(sys.argv[1]))
tail = {int(seed) for seed in payload["unsupported_tail_0_of_8"]}
hard = sorted(
    int(seed)
    for seed, info in payload["seeds"].items()
    if info["group"] == "hard" and int(seed) not in tail
)
print("\n".join(str(seed) for seed in hard))
EOF
)

echo "Run dir: ${RUN_DIR}"
echo "D_H seed workers: ${#DH_SEEDS[@]}  GPUs: ${GPU_IDS[*]}"

COMMON=(
  --task "${TASK}"
  --task-config demo_clean
  --ckpt-path "${CKPT}"
  --run-dir "${RUN_DIR}"
)

launch_one() {
  local gpu="$1"
  local log="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" "${CT_DIR}/collect_t1c_sources.py" "$@" \
      >> "${log}" 2>&1 &
  PIDS+=($!)
  echo "  GPU ${gpu}: $*  (log: ${log})"
}

PIDS=()
WORKER=0
assign_gpu() {
  echo "${GPU_IDS[$(( WORKER % ${#GPU_IDS[@]} ))]}"
}

for seed in "${DH_SEEDS[@]}"; do
  gpu="$(assign_gpu)"
  log="${RUN_DIR}/D_H.seed_${seed}.log"
  resume=()
  [[ -f "${RUN_DIR}/manifest_D_H.seed_${seed}.jsonl" ]] && resume=(--resume)
  launch_one "${gpu}" "${log}" --group D_H --only-seeds "${seed}" "${resume[@]}" "${COMMON[@]}"
  WORKER=$((WORKER + 1))
done

for group in D_E D_M; do
  gpu="$(assign_gpu)"
  log="${RUN_DIR}/${group}.log"
  resume=()
  [[ -f "${RUN_DIR}/manifest_${group}.jsonl" ]] && resume=(--resume)
  launch_one "${gpu}" "${log}" --group "${group}" "${resume[@]}" "${COMMON[@]}"
  WORKER=$((WORKER + 1))
done

echo "Launched ${#PIDS[@]} workers"

FAIL=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || FAIL=1
done
if [[ "${FAIL}" -ne 0 ]]; then
  echo "One or more workers failed; re-run the same command to resume." >&2
  exit 1
fi

"${PYTHON_BIN}" "${CT_DIR}/collect_t1c_sources.py" \
  --task "${TASK}" --ckpt-path "${CKPT}" --run-dir "${RUN_DIR}" \
  --merge-dh-shards

"${PYTHON_BIN}" "${CT_DIR}/collect_t1c_sources.py" \
  --task "${TASK}" --ckpt-path "${CKPT}" --run-dir "${RUN_DIR}" \
  --report
echo "T1c collection finished: ${RUN_DIR}/t1c_collection_report.json"
