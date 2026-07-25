#!/bin/bash
set -euo pipefail

task_name=${1:?task name required}
main_id=${2:?main id required e.g. A1}
gpu_id=${3:?GPU id required}
train_seed=${4:-0}
epochs=${5:-5}
action_dim=${6:-14}
steps_per_epoch=${7:-}
learning_rate=${8:-1e-5}
expert_ratio=${9:-none}
dataset_suffix=${10:-}
checkpoint_every=${11:-5}
loss_mode=${12:-}
lambda_expert=${13:-}
lambda_rollout=${14:-}
lambda_prefix=${15:-}
normalizer_source=${16:-checkpoint}
rollout_per_batch=${17:-}
prefix_per_batch=${18:-}
group_stratified_rollout=${19:-}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dp_dir="${repo_root}/policy/DP"

mapfile -t resolved < <(
  python "${repo_root}/experiments/phase3/resolve_train_config.py" \
    --task "${task_name}" --main-id "${main_id}"
)
if (( ${#resolved[@]} != 9 )); then
  echo "Failed to resolve canonical Phase 3 settings for ${task_name}/${main_id}" >&2
  exit 1
fi
steps_per_epoch=${steps_per_epoch:-${resolved[0]}}
dataset_suffix=${dataset_suffix:-${resolved[1]}}
loss_mode=${loss_mode:-${resolved[2]}}
lambda_expert=${lambda_expert:-${resolved[3]}}
lambda_rollout=${lambda_rollout:-${resolved[4]}}
lambda_prefix=${lambda_prefix:-${resolved[5]}}
rollout_per_batch=${rollout_per_batch:-${resolved[6]}}
prefix_per_batch=${prefix_per_batch:-${resolved[7]}}
group_stratified_rollout=${group_stratified_rollout:-${resolved[8]}}

checkpoint_name="${task_name}-phase3-${main_id}"
checkpoint_dir="${dp_dir}/checkpoints/${checkpoint_name}-${train_seed}"
base_checkpoint="${dp_dir}/checkpoints/${task_name}-demo_clean-200-0/600.ckpt"
dataset_path="data_phase3/${task_name}-${dataset_suffix}.zarr"

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
  while IFS= read -r candidate_ckpt; do
    epoch_name="$(basename "${candidate_ckpt}" .ckpt)"
    if [[ "${epoch_name}" =~ ^[0-9]+$ ]] && (( epoch_name < epochs )); then
      resume_checkpoint="${candidate_ckpt}"
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
  training.normalizer_source="${normalizer_source}"
  training.loss_mode="${loss_mode}"
  training.lambda_expert="${lambda_expert}"
  training.lambda_rollout="${lambda_rollout}"
  training.lambda_prefix="${lambda_prefix}"
  training.log_source_loss_every=50
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
  echo "Resuming ${task_name}/${main_id}/seed${train_seed} from ${resume_checkpoint}"
fi
if [[ -n "${steps_per_epoch}" ]]; then
  train_args+=(dataloader.num_batches="${steps_per_epoch}")
fi
if [[ "${rollout_per_batch}" != "none" ]]; then
  train_args+=(
    dataloader.rollout_per_batch="${rollout_per_batch}"
    dataloader.prefix_per_batch="${prefix_per_batch}"
    dataloader.group_stratified_rollout="${group_stratified_rollout}"
    dataloader.expert_ratio=null
  )
elif [[ "${expert_ratio}" == "none" ]]; then
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
