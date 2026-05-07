#!/usr/bin/env bash
# Copy RDT processed_data into training_data/<run_name> for finetune.
# Usage: bash scripts/copy_processed_to_training_data.sh <run_name> <processed_rel_path> [extra_processed_paths...]
# Example:
#   bash scripts/copy_processed_to_training_data.sh cup_base_170m processed_data/lift_empty_cup-demo_clean-100
# Mix datasets: pass multiple processed paths; duplicate a path with different dest names to up-weight sampling.
set -euo pipefail
RUN_NAME="${1:?run_name}"
shift
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_name> <processed_dir> [more_processed_dirs...]" >&2
  exit 1
fi
DEST_ROOT="training_data/${RUN_NAME}"
mkdir -p "${DEST_ROOT}"
for src in "$@"; do
  base="$(basename "${src}")"
  cp -r "${src}" "${DEST_ROOT}/${base}"
  echo "Copied ${src} -> ${DEST_ROOT}/${base}"
done
