#!/bin/bash
set -euo pipefail

task=${1:?task}
method=${2:?method}
gpu=${3:?gpu}
train_seed=${4:?train seed}
epochs=${5:?target epoch}
dataset_path=${6:?screen zarr path}
checkpoint_label=${7:?checkpoint label}
steps_per_epoch=${8:?steps per epoch}
learning_rate=${9:-5e-5}
rollout_per_batch=${10:-16}
batch_size=${11:-128}
anchor_zarr_path=${12:-}
screen_protocol_path=${13:-${BRACE_SCREEN_PROTOCOL_PATH:-}}

case "${method}" in
  B1|N1|U1) anchor=false ;;
  B2|B3) anchor=true ;;
  *) echo "unsupported BRACE method: ${method}" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dp_dir="${repo_root}/policy/DP"
base_checkpoint="${dp_dir}/checkpoints/${task}-demo_clean-200-0/600.ckpt"
checkpoint_name="${task}-brace-${checkpoint_label}-${method}"
checkpoint_dir="${dp_dir}/checkpoints/${checkpoint_name}-${train_seed}"

[[ -s "${base_checkpoint}" ]] || { echo "missing base checkpoint: ${base_checkpoint}" >&2; exit 2; }
[[ -d "${dataset_path}" ]] || { echo "missing screen dataset: ${dataset_path}" >&2; exit 2; }
if [[ "${anchor}" == "true" ]]; then
  [[ -n "${anchor_zarr_path}" && -d "${anchor_zarr_path}" ]] || {
    echo "missing anchor replay zarr for ${method}: ${anchor_zarr_path}" >&2
    exit 2
  }
fi

resume_checkpoint=""
if [[ -d "${checkpoint_dir}" ]]; then
  while IFS= read -r candidate; do
    stem="$(basename "${candidate}" .ckpt)"
    if [[ "${stem}" =~ ^[0-9]+$ ]] && (( stem < epochs )); then
      resume_checkpoint="${candidate}"
    fi
  done < <(find "${checkpoint_dir}" -maxdepth 1 -type f -name '*.ckpt' | sort -V)
fi

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTHONPATH="${repo_root}:${dp_dir}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${dp_dir}"
args=(
  python train.py --config-name=robot_dp_14.yaml
  task.name="${task}"
  task.dataset.zarr_path="${dataset_path}"
  task.dataset.load_to_memory=False
  dataloader.batch_size="${batch_size}"
  dataloader.num_batches="${steps_per_epoch}"
  dataloader.rollout_per_batch="${rollout_per_batch}"
  training.debug=False
  training.seed="${train_seed}"
  training.device=cuda:0
  training.resume=False
  training.resume_from_ckpt="${base_checkpoint}"
  training.resume_training_ckpt=null
  training.checkpoint_name="${checkpoint_name}"
  training.num_epochs=10
  training.stop_after_epoch="${epochs}"
  training.checkpoint_every=1
  training.normalizer_source=checkpoint
  training.loss_mode=pooled
  training.brace_anchor.enabled="${anchor}"
  optimizer.lr="${learning_rate}"
  exp_name="${checkpoint_name}"
  logging.mode=offline
  setting=demo_clean
  expert_data_num=200
  head_camera_type=D435
)
if [[ "${anchor}" == "true" ]]; then
  args+=(training.brace_anchor.dataset.zarr_path="${anchor_zarr_path}")
fi
if [[ -n "${resume_checkpoint}" ]]; then
  args+=(training.resume_from_ckpt=null training.resume_training_ckpt="${resume_checkpoint}")
  echo "Resuming ${method} from ${resume_checkpoint}"
fi
"${args[@]}"

final="${checkpoint_dir}/${epochs}.ckpt"
[[ -s "${final}" ]] || { echo "training exited without ${final}" >&2; exit 1; }
touch "${final}.complete"

if [[ "${anchor}" == "true" ]]; then
  protocol_path="${screen_protocol_path:-${repo_root}/experiments/brace/screen_protocol.v1.2.json}"
  [[ -f "${protocol_path}" ]] || { echo "missing screen protocol: ${protocol_path}" >&2; exit 2; }
  feasibility_json="${checkpoint_dir}/${epochs}.feasibility.json"
  log_path=""
  while IFS= read -r candidate; do
    [[ -f "${candidate}" ]] || continue
    if grep -q "${checkpoint_name}" "$(dirname "${candidate}")/.hydra/config.yaml" 2>/dev/null; then
      log_path="${candidate}"
    fi
  done < <(find "${dp_dir}/data/outputs" -type f -name logs.json.txt 2>/dev/null | sort)
  if [[ -z "${log_path}" ]]; then
    echo "missing training log for anchor feasibility: ${checkpoint_name}" >&2
    exit 1
  fi
  set +e
  PYTHONPATH="${repo_root}" python "${repo_root}/experiments/brace/anchor_feasibility_test.py" \
    --protocol "${protocol_path}" \
    --log "${log_path}" \
    --output "${feasibility_json}"
  analyzer_exit=$?
  set -e
  if [[ ! -s "${feasibility_json}" ]]; then
    echo "anchor feasibility analyzer failed without writing ${feasibility_json}" >&2
    exit 1
  fi
  if (( analyzer_exit != 0 )); then
    echo "anchor constraint feasibility not met; checkpoint marked ineligible: ${feasibility_json}" >&2
  fi
fi
