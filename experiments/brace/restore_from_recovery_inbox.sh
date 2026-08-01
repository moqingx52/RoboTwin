#!/usr/bin/env bash
# Restore BRACE evidence from records/recovery_inbox/ to canonical paths.
# Skips place replay summary.json (failed extract; re-run audit-v2 instead).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

inbox=experiments/brace/records/recovery_inbox
if [[ ! -d "${inbox}" ]]; then
  echo "Missing ${inbox}. Extract recovery_inbox.tar.gz first:" >&2
  echo "  tar xzf recovery_inbox.tar.gz -C experiments/brace/records/" >&2
  exit 2
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
backup=experiments/brace/records/recovery_backup_${stamp}
mkdir -p "${backup}"

copy_with_backup() {
  local src=$1
  local dest=$2
  if [[ ! -f "${src}" ]]; then
    echo "SKIP missing source: ${src}" >&2
    return 0
  fi
  mkdir -p "$(dirname "${dest}")"
  if [[ -f "${dest}" ]]; then
    mkdir -p "${backup}/$(dirname "${dest}")"
    cp -a "${dest}" "${backup}/${dest}"
  fi
  cp -a "${src}" "${dest}"
  echo "RESTORED ${dest}"
}

echo "=== Backups (if any) -> ${backup} ==="

# Branch archives (full recovery)
for bundle in \
  branches_place_pilot_valid_v2.3 \
  branches_dump_pilot_valid_v2.3 \
  branches_place_confirm_v2.3; do
  for name in summary.json checks.jsonl merged_gate.json confirm_seeds.json; do
  src="${inbox}/archive/${bundle}/${name}"
  if [[ -f "${src}" ]]; then
    copy_with_backup "${src}" "experiments/brace/archive/${bundle}/${name}"
  fi
  done
done

# Dump replay gate: summary only (checks missing on cloud)
copy_with_backup \
  "${inbox}/archive/replay_audit_v2_dump_v2.3_gate/summary.json" \
  experiments/brace/archive/replay_audit_v2_dump_v2.3_gate/summary.json

# Place replay gate: auxiliary jsonl only — NOT failed summary
for name in checks.jsonl failures.jsonl diagnostics.jsonl; do
  copy_with_backup \
    "${inbox}/archive/replay_audit_v2_place_v2.3_gate/${name}" \
    "experiments/brace/archive/replay_audit_v2_place_v2.3_gate/${name}"
done

# Datasets
for name in place_pilot_v2.3_B1.jsonl place_pilot_v2.3_N1.jsonl place_pilot_v2.3_summary.json; do
  copy_with_backup "${inbox}/datasets/${name}" "experiments/brace/datasets/${name}"
done

# Budgets into archive bundles when present
for task_file in place_container_plate_pilot.json dump_bin_bigbin_pilot.json; do
  src="${inbox}/budgets/${task_file}"
  if [[ -f "${src}" ]]; then
    if [[ "${task_file}" == place_* ]]; then
      copy_with_backup "${src}" experiments/brace/archive/branches_place_pilot_valid_v2.3/budgets.json
    else
      copy_with_backup "${src}" experiments/brace/archive/branches_dump_pilot_valid_v2.3/budgets.json
    fi
  fi
done

# Protocol copies (canonical)
proto=experiments/brace/protocol.v2.3.json
for bundle in \
  branches_place_pilot_valid_v2.3 \
  branches_dump_pilot_valid_v2.3 \
  branches_place_confirm_v2.3; do
  if [[ -f "${proto}" ]]; then
    copy_with_backup "${proto}" "experiments/brace/archive/${bundle}/protocol.v2.3.json"
  fi
done

# Confirm seeds symlink target
if [[ -f "${inbox}/archive/branches_place_confirm_v2.3/confirm_seeds.json" ]]; then
  mkdir -p experiments/brace/seeds
  copy_with_backup \
    "${inbox}/archive/branches_place_confirm_v2.3/confirm_seeds.json" \
    experiments/brace/seeds/place_container_plate_confirm_seeds.json
fi

# Regenerate MANIFEST.sha256 for branch archives
for bundle in \
  branches_place_pilot_valid_v2.3 \
  branches_dump_pilot_valid_v2.3 \
  branches_place_confirm_v2.3; do
  dir="experiments/brace/archive/${bundle}"
  if [[ -d "${dir}" ]]; then
    manifest_files=()
    for candidate in checks.jsonl summary.json protocol.v2.3.json budgets.json merged_gate.json confirm_seeds.json; do
      [[ -f "${dir}/${candidate}" ]] && manifest_files+=("${dir}/${candidate}")
    done
    if ((${#manifest_files[@]} > 0)); then
      sha256sum "${manifest_files[@]}" > "${dir}/MANIFEST.sha256"
      echo "MANIFEST ${dir}/MANIFEST.sha256"
    fi
  fi
done

# Recovery provenance note (not a promote-run source_run; marks manual recovery)
python - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path

repo = Path("experiments/brace")
note = {
    "schema_version": 1,
    "recovered_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "source": "records/recovery_inbox",
    "note": "Manual recovery from cloud inbox. Re-run audit-v2 for place/dump replay gate summaries.",
    "skipped": ["archive/replay_audit_v2_place_v2.3_gate/summary.json (failed extract)"],
}
for bundle in [
    "branches_place_pilot_valid_v2.3",
    "branches_dump_pilot_valid_v2.3",
    "branches_place_confirm_v2.3",
]:
    path = repo / "archive" / bundle / "source_run.json"
    if path.parent.is_dir() and not path.exists():
        path.write_text(json.dumps({**note, "target": f"archive/{bundle}"}, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote provenance {path}")
PY

echo ""
echo "=== Restore complete ==="
echo "NOT restored (re-run required):"
echo "  - archive/replay_audit_v2_place_v2.3_gate/summary.json"
echo "  - archive/replay_audit_v2_dump_v2.3_gate/checks.jsonl (+ failures/diagnostics)"
echo ""
echo "Verify:"
echo "  sha256sum experiments/brace/archive/branches_place_pilot_valid_v2.3/summary.json"
echo "  # expect 8ef8d3585c23fcde6b53cce441e4ca8b939dab5038c65d5a968caf75f2266037"
