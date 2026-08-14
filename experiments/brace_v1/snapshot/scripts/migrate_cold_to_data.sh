#!/usr/bin/env bash
# Migrate cold BRACE artifacts from /depot to /data with symlink preservation.
# Usage: migrate_cold_to_data.sh [--dry-run] <target>
# Targets: cleanup-pre-v2 | cleanup-base200-intermediate | run-confirmatory | run-calibration | rollouts-traced
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

REPO="/depot/rlinf/repos/RoboTwin"
COLD_ROOT="/data/rlinf_archive/brace-cold"
LOG_DIR="${COLD_ROOT}/logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date -Iseconds)] $*"; }

run_cmd() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY-RUN: $*"
  else
    "$@"
  fi
}

inventory_dir() {
  local src="$1"
  local label="$2"
  local out="${LOG_DIR}/${label}_inventory.txt"
  log "inventory -> $out"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY-RUN inventory $src"
    return 0
  fi
  {
    echo "# inventory for $src"
    echo "# generated $(date -Iseconds)"
    find "$src" -type f -printf '%P\t%s\n' | sort
    echo "# total_bytes $(find "$src" -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')"
    echo "# file_count $(find "$src" -type f | wc -l)"
  } > "$out"
}

verify_copy() {
  local src="$1"
  local dst="$2"
  local src_count src_bytes dst_count dst_bytes
  src_count=$(find "$src" -type f | wc -l)
  dst_count=$(find "$dst" -type f | wc -l)
  src_bytes=$(find "$src" -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')
  dst_bytes=$(find "$dst" -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')
  if [[ "$src_count" != "$dst_count" || "$src_bytes" != "$dst_bytes" ]]; then
    log "VERIFY FAIL: $src -> $dst (files $src_count/$dst_count bytes $src_bytes/$dst_bytes)"
    return 1
  fi
  log "VERIFY OK: $src -> $dst ($src_count files, $src_bytes bytes)"
}

migrate_tree() {
  local src="$1"
  local dst="$2"
  local label="$3"
  if [[ ! -e "$src" ]]; then
    log "skip missing $src"
    return 0
  fi
  if mountpoint -q "$src" 2>/dev/null; then
    log "already bind-mounted: $src"
    return 0
  fi
  if [[ -L "$src" ]]; then
    log "already symlink: $src -> $(readlink -f "$src")"
    return 0
  fi
  inventory_dir "$src" "$label"
  log "rsync $src -> $dst"
  run_cmd mkdir -p "$(dirname "$dst")"
  run_cmd rsync -aH --info=progress2 "$src/" "$dst/"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  verify_copy "$src" "$dst"
  log "host bind mount $src -> $dst"
  run_cmd docker run --rm --privileged --pid=host alpine:3.19 sh -c "
    set -e
    nsenter -t 1 -m sh -c 'test -d \"$src\" || mkdir -p \"$src\"'
    nsenter -t 1 -m sh -c 'mountpoint -q \"$src\" && umount \"$src\" || true'
    nsenter -t 1 -m mount --bind \"$dst\" \"$src\"
    nsenter -t 1 -m mountpoint \"$src\"
  "
}

cleanup_pre_v2() {
  local removed=0
  mapfile -t items < <(find "$REPO/policy/DP" \( -name '*.pre_v2_*' -o -path '*/checkpoints/*.pre_v2_*' \) 2>/dev/null | sort)
  for p in "${items[@]}"; do
    [[ -e "$p" ]] || continue
    # Require matching formal zarr/checkpoint without pre_v2 suffix.
    local formal="${p%%.pre_v2_*}"
    if [[ -e "$formal" ]]; then
      log "remove pre_v2 backup: $p ($(du -sh "$p" | cut -f1))"
      run_cmd rm -rf "$p"
      removed=$((removed + 1))
    else
      log "keep pre_v2 (no formal): $p"
    fi
  done
  log "removed $removed pre_v2 items"
}

cleanup_base200_intermediate() {
  local ckpt_root="$REPO/policy/DP/checkpoints"
  local tasks=(
    beat_block_hammer click_alarmclock handover_mic lift_pot move_can_pot
    open_laptop place_burger_fries place_container_plate put_object_cabinet shake_bottle
  )
  for task in "${tasks[@]}"; do
    local d="${ckpt_root}/${task}-demo_clean-200-0"
    [[ -d "$d" ]] || continue
    [[ -f "$d/600.ckpt" ]] || { log "skip $task (no 600.ckpt)"; continue; }
    [[ -f "$d/brace_base200_provenance.json" ]] || { log "skip $task (no provenance)"; continue; }
    for epoch in 100 200 300 400 500; do
      local f="${d}/${epoch}.ckpt"
      if [[ -f "$f" ]]; then
        log "remove intermediate $f ($(du -sh "$f" | cut -f1))"
        run_cmd rm -f "$f"
      fi
    done
  done
  log "stack_bowls_three left untouched (in-flight Base200)"
}

target="${1:-}"
case "$target" in
  cleanup-pre-v2) cleanup_pre_v2 ;;
  cleanup-base200-intermediate) cleanup_base200_intermediate ;;
  run-confirmatory)
    migrate_tree \
      "$REPO/experiments/brace/runs/20260804T213700Z_confirmatory_preservation_place_container_plate_dump_bin_bigbin" \
      "$COLD_ROOT/runs/20260804T213700Z_confirmatory_preservation_place_container_plate_dump_bin_bigbin" \
      "run_confirmatory_preservation"
    ;;
  run-calibration)
    migrate_tree \
      "$REPO/experiments/brace/runs/20260803T031036Z_anchor_calibration_place_container_plate" \
      "$COLD_ROOT/runs/20260803T031036Z_anchor_calibration_place_container_plate" \
      "run_anchor_calibration"
    ;;
  rollouts-traced)
    migrate_tree \
      "$REPO/experiments/brace/rollouts_traced" \
      "$COLD_ROOT/rollouts_traced" \
      "rollouts_traced"
    ;;
  *)
    echo "Unknown target: $target" >&2
    exit 2
    ;;
esac
