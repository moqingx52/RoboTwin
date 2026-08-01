#!/usr/bin/env bash
# P0: freeze place pilot branch artifacts with SHA256 manifest.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

archive_dir=experiments/brace/archive/branches_place_pilot_valid_v2.3
mkdir -p "${archive_dir}"

required=(
  experiments/brace/protocol.v2.3.json
)
optional=(
  experiments/brace/branches/checks.jsonl
  experiments/brace/budgets/place_container_plate_pilot.json
  experiments/brace/branches/summary.json
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
    cp "${src}" "${archive_dir}/"
  else
    echo "Warning: optional artifact not found (copy from cloud if missing): ${src}" >&2
  fi
done

if [[ ! -f "${archive_dir}/summary.json" ]]; then
  echo "Missing archived summary.json in ${archive_dir}" >&2
  exit 2
fi

manifest_files=()
for candidate in checks.jsonl summary.json protocol.v2.3.json; do
  if [[ -f "${archive_dir}/${candidate}" ]]; then
    manifest_files+=("${archive_dir}/${candidate}")
  fi
done

sha256sum "${manifest_files[@]}" > "${archive_dir}/MANIFEST.sha256"

if [[ -f experiments/brace/replay_audit_v2/summary.json ]]; then
  dump_archive=experiments/brace/archive/replay_audit_v2_dump_v2.3_gate
  mkdir -p "${dump_archive}"
  if [[ ! -f "${dump_archive}/summary.json" ]]; then
    jq 'del(.tasks.place_container_plate) | .tasks |= with_entries(select(.key == "dump_bin_bigbin"))' \
      experiments/brace/replay_audit_v2/summary.json > "${dump_archive}/summary.json" || true
  fi
fi

echo "Archived to ${archive_dir}"
cat "${archive_dir}/MANIFEST.sha256"
