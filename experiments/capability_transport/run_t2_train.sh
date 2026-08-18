#!/usr/bin/env bash
# T2 matched small-dose training: 6 points × K=12 seeds = 72 DP jobs.
# Frozen config: experiments/capability_transport/t2_launch_config.json
#
# One trainer per GPU. Does not colocate simulator workers. Resume in place.
#
# Usage (inside the cloud container, from /workspace/RoboTwin):
#   bash experiments/capability_transport/run_t2_train.sh --dry-run
#   bash experiments/capability_transport/run_t2_train.sh --build-only
#   bash experiments/capability_transport/run_t2_train.sh
#   bash experiments/capability_transport/run_t2_train.sh --run-dir /path/to/run
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CT_DIR="${REPO_ROOT}/experiments/capability_transport"

if [[ -n "${T2_TRAIN_PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${T2_TRAIN_PYTHON_BIN}"
elif [[ -x /root/miniconda/envs/RoboTwin/bin/python ]]; then
  PYTHON_BIN="/root/miniconda/envs/RoboTwin/bin/python"
else
  PYTHON_BIN="python"
fi

export T2_TRAIN_GPU_IDS="${T2_TRAIN_GPU_IDS:-0 1 2 3 4 5 6 7}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" "${CT_DIR}/launch_t2_train.py" "$@"
