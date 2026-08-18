#!/usr/bin/env bash
# T2 independent eval census: 200 seeds x R=8 baseline rollouts of pi0.
# Protocol: experiments/capability_transport/protocol.t2_eval_census.v1.json
#
# Launches 24 eval_per_seed.py shards (3 per GPU on 8 GPUs), each with its own
# CUDA_VISIBLE_DEVICES, all with --resume; merges shards on success.
#
# Usage (inside the cloud container, from /workspace/RoboTwin):
#   bash experiments/capability_transport/run_t2_eval_census.sh place_container_plate
#   bash experiments/capability_transport/run_t2_eval_census.sh place_container_plate /path/to/run_dir
#
# Re-running the same command resumes an interrupted run in place.
set -euo pipefail

TASK="${1:?usage: run_t2_eval_census.sh <task> [run_dir]}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CT_DIR="${REPO_ROOT}/experiments/capability_transport"

GPU_IDS=(${T2_EVAL_GPU_IDS:-0 1 2 3 4 5 6 7})
WORKERS_PER_GPU="${T2_EVAL_WORKERS_PER_GPU:-3}"
NUM_SHARDS=$(( ${#GPU_IDS[@]} * WORKERS_PER_GPU ))
if [[ -n "${T2_EVAL_PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${T2_EVAL_PYTHON_BIN}"
elif [[ -x /root/miniconda/envs/RoboTwin/bin/python ]]; then
  PYTHON_BIN="/root/miniconda/envs/RoboTwin/bin/python"
else
  PYTHON_BIN="python"
fi

CKPT="${REPO_ROOT}/policy/DP/checkpoints/${TASK}-demo_clean-200-0/600.ckpt"
SPLITS_FILE="${CT_DIR}/seeds/${TASK}_t2_eval_census_splits.json"
SEEDS_FILE="${CT_DIR}/seeds/${TASK}_seeds.json"
POLICY_SEED_OFFSET=7000
REPEATS=8
VARIANT="t2_eval_census_v1"

if [[ $# -ge 2 ]]; then
  RUN_DIR="$2"
else
  LATEST="${CT_DIR}/runs/LATEST_T2_EVAL_CENSUS_${TASK}"
  if [[ -f "${LATEST}" ]] && [[ -d "$(cat "${LATEST}")" ]]; then
    RUN_DIR="$(cat "${LATEST}")"
  else
    RUN_DIR="${CT_DIR}/runs/$(date -u +%Y%m%dT%H%M%SZ)_t2_eval_census_${TASK}"
  fi
fi
mkdir -p "${RUN_DIR}"
echo "${RUN_DIR}" > "${CT_DIR}/runs/LATEST_T2_EVAL_CENSUS_${TASK}"

[[ -f "${CKPT}" ]] || { echo "missing checkpoint: ${CKPT}" >&2; exit 1; }
[[ -f "${SPLITS_FILE}" ]] || { echo "missing splits file: ${SPLITS_FILE}" >&2; exit 1; }
[[ -f "${CT_DIR}/protocol.t2_eval_census.v1.json" ]] || {
  echo "missing frozen protocol: ${CT_DIR}/protocol.t2_eval_census.v1.json" >&2
  exit 1
}

COMMON_ARGS=(
  --task "${TASK}"
  --task-config demo_clean
  --variant "${VARIANT}"
  --ckpt-path "${CKPT}"
  --seeds-file "${SEEDS_FILE}"
  --no-include-hard
  --id-repeats 0
  --train-repeats 0
  --extra-splits-file "${SPLITS_FILE}"
  --extra-split-repeats "${REPEATS}"
  --policy-seed-offset "${POLICY_SEED_OFFSET}"
  --output-dir "${RUN_DIR}"
)

echo "Run dir: ${RUN_DIR}"
echo "Protocol: protocol.t2_eval_census.v1.json"
echo "Shards: ${NUM_SHARDS} (${#GPU_IDS[@]} GPUs x ${WORKERS_PER_GPU})"

PIDS=()
for (( shard=0; shard<NUM_SHARDS; shard++ )); do
  gpu="${GPU_IDS[$(( shard / WORKERS_PER_GPU ))]}"
  log="${RUN_DIR}/shard_$(printf '%02d' "${shard}").log"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" "${CT_DIR}/eval_per_seed.py" \
      "${COMMON_ARGS[@]}" \
      --shard-id "${shard}" --num-shards "${NUM_SHARDS}" \
      --resume \
      >> "${log}" 2>&1 &
  PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || FAIL=1
done
if [[ "${FAIL}" -ne 0 ]]; then
  echo "One or more shards failed; re-run the same command to resume." >&2
  exit 1
fi

"${PYTHON_BIN}" "${CT_DIR}/merge_eval_shards.py" \
  --task "${TASK}" --task-config demo_clean \
  --variant "${VARIANT}" \
  --output-dir "${RUN_DIR}" \
  --num-shards "${NUM_SHARDS}"

"${PYTHON_BIN}" "${CT_DIR}/eval_per_seed.py" "${COMMON_ARGS[@]}" --check-complete-result
echo "T2 eval census complete: ${RUN_DIR}/${TASK}/${VARIANT}.json"
echo "Next: python experiments/capability_transport/make_t2_eval_panels.py --result ${RUN_DIR}/${TASK}/${VARIANT}.json"
