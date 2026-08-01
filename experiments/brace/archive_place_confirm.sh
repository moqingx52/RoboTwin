#!/usr/bin/env bash
# P0: freeze place confirmatory branch artifacts with SHA256 manifest.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

archive_dir=experiments/brace/archive/branches_place_confirm_v2.3
branch_dir=experiments/brace/branches_confirm
pilot_archive=experiments/brace/archive/branches_place_pilot_valid_v2.3

mkdir -p "${archive_dir}"

if [[ ! -f "${branch_dir}/summary.json" ]]; then
  echo "Missing confirm branch summary: ${branch_dir}/summary.json" >&2
  exit 2
fi

python experiments/brace/evaluate_confirmatory_gate.py \
  --pilot-summary "${pilot_archive}/summary.json" \
  --confirm-summary "${branch_dir}/summary.json" \
  --output "${archive_dir}/merged_gate.json" || true

required=(experiments/brace/protocol.v2.3.json)
optional=(
  "${branch_dir}/checks.jsonl"
  "${branch_dir}/summary.json"
  experiments/brace/seeds/place_container_plate_confirm_seeds.json
)

for src in "${required[@]}"; do
  if [[ ! -f "${src}" ]]; then
    echo "Missing required artifact: ${src}" >&2
    exit 2
  fi
  cp "${src}" "${archive_dir}/"
done

for src in "${optional[@]}"; do
  if [[ -f "${src}" ]]; then
    dest_name="$(basename "${src}")"
    if [[ "${src}" == *confirm_seeds.json ]]; then
      dest_name="confirm_seeds.json"
    fi
    cp "${src}" "${archive_dir}/${dest_name}"
  else
    echo "Warning: optional artifact not found: ${src}" >&2
  fi
done

if [[ ! -f "${archive_dir}/merged_gate.json" ]]; then
  echo "Warning: merged_gate.json not created; evaluate_confirmatory_gate may have failed." >&2
fi

manifest_files=()
for candidate in checks.jsonl summary.json protocol.v2.3.json merged_gate.json confirm_seeds.json; do
  if [[ -f "${archive_dir}/${candidate}" ]]; then
    manifest_files+=("${archive_dir}/${candidate}")
  fi
done

if [[ ${#manifest_files[@]} -eq 0 ]]; then
  echo "No files to manifest in ${archive_dir}" >&2
  exit 2
fi

sha256sum "${manifest_files[@]}" > "${archive_dir}/MANIFEST.sha256"
echo "Archived to ${archive_dir}"
cat "${archive_dir}/MANIFEST.sha256"
