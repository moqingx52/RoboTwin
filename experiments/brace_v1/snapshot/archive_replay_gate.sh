#!/usr/bin/env bash
# Freeze per-task replay audit v2 gate summary into archive/.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

task="${BRACE_ARCHIVE_TASK:-place_container_plate}"
audit_root="${BRACE_AUDIT_ROOT:-experiments/brace/replay_audit_v2}"

python experiments/brace/extract_replay_gate_archive.py \
  --task "${task}" \
  --audit-root "${audit_root}"
