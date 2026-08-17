#!/usr/bin/env bash
# Resume D_M (or D_E) with round-parallel workers: 3 per GPU on 8 GPUs.
# Official manifest stays serial earliest-stop (commit in frozen seed order).
#
# Usage (container, after stopping the serial D_M worker):
#   bash experiments/capability_transport/run_t1c_dm_parallel.sh place_container_plate <run_dir>
set -euo pipefail

TASK="${1:?usage: run_t1c_dm_parallel.sh <task> <run_dir>}"
RUN_DIR="${2:?usage: run_t1c_dm_parallel.sh <task> <run_dir>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CT_DIR="${REPO_ROOT}/experiments/capability_transport"
PYTHON_BIN="${T1_PYTHON_BIN:-python}"
CKPT="${REPO_ROOT}/policy/DP/checkpoints/${TASK}-demo_clean-200-0/600.ckpt"

echo "Parallel D_M resume: ${RUN_DIR}"
"${PYTHON_BIN}" "${CT_DIR}/collect_t1c_sources.py" \
  --task "${TASK}" \
  --task-config demo_clean \
  --group D_M \
  --ckpt-path "${CKPT}" \
  --run-dir "${RUN_DIR}" \
  --parallel-rounds \
  --gpu-ids "${T1_GPU_IDS:-0 1 2 3 4 5 6 7}" \
  --workers-per-gpu "${T1_WORKERS_PER_GPU:-3}"

"${PYTHON_BIN}" "${CT_DIR}/collect_t1c_sources.py" \
  --task "${TASK}" --ckpt-path "${CKPT}" --run-dir "${RUN_DIR}" \
  --merge-dh-shards

"${PYTHON_BIN}" "${CT_DIR}/collect_t1c_sources.py" \
  --task "${TASK}" --ckpt-path "${CKPT}" --run-dir "${RUN_DIR}" \
  --report
echo "T1c collection finished: ${RUN_DIR}/t1c_collection_report.json"
