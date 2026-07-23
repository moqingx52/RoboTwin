#!/bin/bash
set -euo pipefail

task_name=${1:?task name required}
variant=${2:?variant required}
gpu_id=${3:?GPU id required}
train_seed=${4:-0}
epochs=${5:-50}
action_dim=${6:-14}
steps_per_epoch=${7:-}
learning_rate=${8:-1e-5}
expert_ratio=${9:-none}
dataset_suffix=${10:-success}
checkpoint_every=${11:-10}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dp_dir="${repo_root}/policy/DP"
checkpoint_name="${task_name}-phase2-${variant}"
checkpoint_dir="${dp_dir}/checkpoints/${checkpoint_name}-${train_seed}"
base_checkpoint="${dp_dir}/checkpoints/${task_name}-demo_clean-200-0/600.ckpt"
dataset_path="data_phase1_200/${task_name}-${dataset_suffix}.zarr"

if [[ ! -s "${base_checkpoint}" ]]; then
  echo "Missing base checkpoint: ${base_checkpoint}" >&2
  exit 1
fi
if [[ ! -d "${dp_dir}/${dataset_path}" ]]; then
  echo "Missing dataset: ${dp_dir}/${dataset_path}" >&2
  exit 1
fi

resume_checkpoint=""
if [[ -d "${checkpoint_dir}" ]]; then
  while IFS= read -r candidate; do
    epoch_name="$(basename "${candidate}" .ckpt)"
    if [[ "${epoch_name}" =~ ^[0-9]+$ ]] && (( epoch_name < epochs )); then
      resume_checkpoint="${candidate}"
    fi
  done < <(find "${checkpoint_dir}" -maxdepth 1 -type f -name '*.ckpt' | sort -V)
fi

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${gpu_id}"
cd "${dp_dir}"

train_args=(
  python train.py --config-name="robot_dp_${action_dim}.yaml"
  task.name="${task_name}"
  task.dataset.zarr_path="${dataset_path}"
  task.dataset.load_to_memory=False
  training.debug=False
  training.seed="${train_seed}"
  training.device="cuda:0"
  training.resume=False
  training.resume_from_ckpt="${base_checkpoint}"
  training.resume_training_ckpt=null
  training.checkpoint_name="${checkpoint_name}"
  training.num_epochs="${epochs}"
  training.checkpoint_every="${checkpoint_every}"
  optimizer.lr="${learning_rate}"
  exp_name="${checkpoint_name}"
  logging.mode=offline
  setting="demo_clean"
  expert_data_num=200
  head_camera_type=D435
)

if [[ -n "${resume_checkpoint}" ]]; then
  train_args+=(
    training.resume_from_ckpt=null
    training.resume_training_ckpt="${resume_checkpoint}"
  )
  echo "Resuming ${task_name}/${variant}/seed${train_seed} from ${resume_checkpoint}"
fi
if [[ -n "${steps_per_epoch}" ]]; then
  train_args+=(dataloader.num_batches="${steps_per_epoch}")
fi
if [[ "${expert_ratio}" == "none" ]]; then
  train_args+=(dataloader.expert_ratio=null)
else
  train_args+=(dataloader.expert_ratio="${expert_ratio}")
fi

"${train_args[@]}"

final_checkpoint="${checkpoint_dir}/${epochs}.ckpt"
if [[ ! -s "${final_checkpoint}" ]]; then
  echo "Training exited without final checkpoint: ${final_checkpoint}" >&2
  exit 1
fi
touch "${final_checkpoint}.complete"
