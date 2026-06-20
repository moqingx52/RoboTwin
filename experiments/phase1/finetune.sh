#!/bin/bash
set -euo pipefail

task_name=${1}
variant=${2}
gpu_id=${3:-0}
train_seed=${4:-0}
epochs=${5:-200}
action_dim=${6:-14}

case "${variant}" in
  success|seed_balanced|difficulty_weighted) ;;
  *)
    echo "variant must be one of: success, seed_balanced, difficulty_weighted" >&2
    exit 1
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dp_dir="${repo_root}/policy/DP"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${gpu_id}"

cd "${dp_dir}"

python train.py --config-name="robot_dp_${action_dim}.yaml" \
  task.name="${task_name}" \
  task.dataset.zarr_path="data/${task_name}-${variant}.zarr" \
  training.debug=False \
  training.seed="${train_seed}" \
  training.device="cuda:0" \
  training.resume=False \
  training.resume_from_ckpt="checkpoints/${task_name}-demo_clean-50-${train_seed}/600.ckpt" \
  training.num_epochs="${epochs}" \
  training.checkpoint_every="${epochs}" \
  optimizer.lr=5e-5 \
  exp_name="${task_name}-robot_dp-phase1-${variant}" \
  logging.mode=offline \
  setting="demo_clean" \
  expert_data_num=50 \
  head_camera_type=D435

