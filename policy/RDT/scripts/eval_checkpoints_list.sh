#!/usr/bin/env bash
# Run eval.sh for several checkpoint ids (e.g. every 2500 steps). Call from policy/RDT.
# Usage: bash scripts/eval_checkpoints_list.sh <task_name> <task_config> <model_name> <seed> <gpu_id> <ckpt_id> [<ckpt_id> ...]
# Set rdt_base_config in policy/RDT/deploy_policy.yml (configs/base_170m.yaml for 170M finetunes).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${HERE}"
TASK_NAME="${1:?task_name}"
TASK_CFG="${2:?task_config}"
MODEL_NAME="${3:?model_name}"
SEED="${4:?seed}"
GPU="${5:?gpu}"
shift 5
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <task> <task_config> <model_name> <seed> <gpu> <ckpt> [ckpt ...]" >&2
  exit 1
fi
for ck in "$@"; do
  echo "========== checkpoint-${ck} =========="
  bash eval.sh "${TASK_NAME}" "${TASK_CFG}" "${MODEL_NAME}" "${ck}" "${SEED}" "${GPU}"
done
