#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

source /root/miniconda/etc/profile.d/conda.sh
conda activate RoboTwin

export BRACE_GPU_IDS="${BRACE_GPU_IDS:-0 1 2 3 4 5 6 7}"
export EVAL_WORKERS_PER_GPU="${EVAL_WORKERS_PER_GPU:-3}"

census_run="${BRACE_CENSUS_DIR:-experiments/brace/runs/20260804T070036Z_confirmatory_base_census_place_container_plate_dump_bin_bigbin}"
if [[ ! -d "${census_run}/place_container_plate" ]]; then
  echo "Missing census run dir: ${census_run}" >&2
  exit 2
fi

log_dir="${repo_root}/experiments/brace/runs"
log_file="${log_dir}/p1_v142_resume_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "${log_file}" > "${log_dir}/LATEST_p1_full_chain_log"

exec > >(tee -a "${log_file}") 2>&1

echo "=== P1a finalize (re-merge + census_summary) $(date -u -Iseconds) ==="
python experiments/phase1/merge_eval_shards.py \
  --task place_container_plate \
  --task-config demo_clean \
  --variant census_base \
  --output-dir "${census_run}" \
  --num-shards 3
python experiments/brace/confirmatory_base_census.py \
  --task place_container_plate \
  --output "${census_run}" \
  --protocol experiments/brace/screen_protocol.v1.4.2.confirmatory_preservation.json \
  --seeds-file experiments/brace/seeds/place_container_plate_confirmatory_v1.4.2_seeds.json \
  --workers-per-gpu "${EVAL_WORKERS_PER_GPU}" \
  --gpu "${BRACE_CENSUS_GPU:-0}"
echo "${census_run}" > experiments/brace/runs/LATEST_confirmatory_base_census
export BRACE_CENSUS_DIR="${census_run}"

echo "=== P1b select-preservation-cohort $(date -u -Iseconds) ==="
bash experiments/brace/run_all.sh select-preservation-cohort place_container_plate

echo "=== P1c confirmatory-preservation $(date -u -Iseconds) ==="
bash experiments/brace/run_all.sh confirmatory-preservation place_container_plate

echo "=== P1d confirmatory-preservation-eval $(date -u -Iseconds) ==="
bash experiments/brace/run_all.sh confirmatory-preservation-eval place_container_plate

echo "=== P1d report $(date -u -Iseconds) ==="
bash experiments/brace/run_all.sh confirmatory-preservation-report place_container_plate

echo "=== DONE $(date -u -Iseconds) ==="
