#!/usr/bin/env bash
# T1c source acquisition: round-robin success-first collection with pi0.
# Protocol: experiments/capability_transport/protocol.t1.v1.2.json
# (D_E/D_M target mode; D_H fixed opportunity budget + joint gate G_H in --report)
#
# The frozen acquisition rule is strictly serial WITHIN a group (round-robin,
# earliest-stop), so each group runs as one process; the three groups are
# independent and run in parallel on three GPUs. Re-running the same command
# resumes in place (append-only manifests; resume replays the frozen traversal
# and continues at the first unrecorded attempt).
#
# Usage (inside the cloud container, from /workspace/RoboTwin):
#   bash experiments/capability_transport/run_t1c_collection.sh place_container_plate
set -euo pipefail

TASK="${1:?usage: run_t1c_collection.sh <task> [run_dir]}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CT_DIR="${REPO_ROOT}/experiments/capability_transport"

GPU_IDS=(${T1_GPU_IDS:-0 1 2})
# Cloud container: /root/miniconda/envs/RoboTwin/bin/python
PYTHON_BIN="${T1_PYTHON_BIN:-python}"

CKPT="${REPO_ROOT}/policy/DP/checkpoints/${TASK}-demo_clean-200-0/600.ckpt"
GROUPS=(D_E D_M D_H)

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

# Resume into an existing run dir if given; otherwise create a UTC-stamped one.
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

echo "Run dir: ${RUN_DIR}"

PIDS=()
for i in "${!GROUPS[@]}"; do
  group="${GROUPS[$i]}"
  gpu="${GPU_IDS[$(( i % ${#GPU_IDS[@]} ))]}"
  log="${RUN_DIR}/${group}.log"
  RESUME_FLAG=()
  [[ -f "${RUN_DIR}/manifest_${group}.jsonl" ]] && RESUME_FLAG=(--resume)
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" "${CT_DIR}/collect_t1c_sources.py" \
      --task "${TASK}" \
      --task-config demo_clean \
      --group "${group}" \
      --ckpt-path "${CKPT}" \
      --run-dir "${RUN_DIR}" \
      "${RESUME_FLAG[@]}" \
      >> "${log}" 2>&1 &
  PIDS+=($!)
  echo "  ${group} -> GPU ${gpu} (log: ${log})"
done

FAIL=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || FAIL=1
done
if [[ "${FAIL}" -ne 0 ]]; then
  echo "One or more groups failed; re-run the same command to resume." >&2
  exit 1
fi

"${PYTHON_BIN}" "${CT_DIR}/collect_t1c_sources.py" \
  --task "${TASK}" \
  --ckpt-path "${CKPT}" \
  --run-dir "${RUN_DIR}" \
  --report
echo "T1c collection finished: ${RUN_DIR}/t1c_collection_report.json"
