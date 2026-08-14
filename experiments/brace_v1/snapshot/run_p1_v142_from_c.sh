#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

source /root/miniconda/etc/profile.d/conda.sh
conda activate RoboTwin

export BRACE_GPU_IDS="${BRACE_GPU_IDS:-0 1 2 3 4 5 6 7}"
export EVAL_WORKERS_PER_GPU="${EVAL_WORKERS_PER_GPU:-3}"
export BRACE_CENSUS_DIR="${BRACE_CENSUS_DIR:-experiments/brace/runs/20260804T070036Z_confirmatory_base_census_place_container_plate_dump_bin_bigbin}"

log_dir="${repo_root}/experiments/brace/runs"
log_file="${log_dir}/p1_v142_from_c_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "${log_file}" > "${log_dir}/LATEST_p1_full_chain_log"

exec > >(tee -a "${log_file}") 2>&1

echo "=== P1c confirmatory-preservation $(date -u -Iseconds) ==="
bash experiments/brace/run_all.sh confirmatory-preservation place_container_plate

echo "=== P1d confirmatory-preservation-eval $(date -u -Iseconds) ==="
bash experiments/brace/run_all.sh confirmatory-preservation-eval place_container_plate

echo "=== P1d report $(date -u -Iseconds) ==="
bash experiments/brace/run_all.sh confirmatory-preservation-report place_container_plate

echo "=== DONE $(date -u -Iseconds) ==="
