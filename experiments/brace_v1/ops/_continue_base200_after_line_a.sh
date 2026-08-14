#!/bin/bash
# Finish remaining Base200 substrate work after Line A stop-line.
# Does not freeze BRACE-v2 or launch held-out confirmatory arms.
set -euo pipefail
cd /workspace/RoboTwin
source /root/miniconda/etc/profile.d/conda.sh
conda activate RoboTwin
export PYTHONPATH="/workspace/RoboTwin:/workspace/RoboTwin/policy/DP${PYTHONPATH:+:${PYTHONPATH}}"

LOG_ROOT=experiments/brace/logs/base200_pipeline_continue
mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${LOG_ROOT}/continue_$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1

echo "=== continue Base200 pipeline $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "will not run method_freeze / heldout_brace / track4"

export BRACE_GPU_IDS="0"
export BRACE_BASE200_STAGE=train
export BRACE_BASE200_TASKS=stack_bowls_three
export BRACE_FAIL_ON_TASK_ERROR=1
export BRACE_BASE200_RUN_DIR="experiments/brace/runs/base200_assets_stack_bowls_$(date -u +%Y%m%dT%H%M%SZ)"
bash experiments/brace/run_multitask_base200_assets.sh

echo "=== stack_bowls train done, starting frozen eval resume ==="
export BRACE_GPU_IDS="0 1 2 3 4 5 6 7"
export BRACE_EVAL_WORKERS_PER_GPU=3
export BRACE_BASE200_EVAL_TASKS="place_container_plate beat_block_hammer click_alarmclock handover_mic lift_pot move_can_pot open_laptop place_burger_fries put_object_cabinet shake_bottle stack_bowls_three"
export BRACE_BASE200_EVAL_RUN_DIR="experiments/brace/runs/base200_frozen_eval_20260811T012027Z"
bash experiments/brace/run_multitask_base200_eval.sh

echo "=== Base200 collect/process/train/eval pipeline complete $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "held-out BRACE not started: Line A joint gate is no-go / freeze_blocked"
