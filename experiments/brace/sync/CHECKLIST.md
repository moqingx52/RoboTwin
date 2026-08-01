# BRACE artifact sync checklist

## Workflow (cloud pull-only, local push)

1. **Cloud** (read-only git): `git pull` → run stages → **promote-run** → `artifact-inventory`
2. **Download** inventory + JSON files to your **local** repo (scp/rsync/tar; see below)
3. **Local**: `validate-artifacts --inventory sync/LATEST` → `git add` → `git commit` → `git push`

Cloud machines typically **cannot push**; this checklist is the handoff manifest.
Mutable working copies under `runs/` are never git-tracked.

Generated: 2026-08-01T18:50:37Z UTC
Host: gsy-ThinkStation-P3-Tower
Git commit: 10f299465e20333e2d9c3a195a05c96c72ca4ff4
Protocol: v2.3

## Promote before inventory

```bash
# After audit-v2 (place example):
BRACE_PROMOTE_RUN=$(cat experiments/brace/runs/LATEST_AUDIT_place_container_plate) \
  BRACE_PROMOTE_TARGET=archive/replay_audit_v2_place_v2.3_gate \
  bash experiments/brace/run_all.sh promote-run

# After branch:
BRACE_PROMOTE_RUN=$(cat experiments/brace/runs/LATEST_branches) \
  BRACE_PROMOTE_TARGET=archive/branches_place_pilot_valid_v2.3 \
  bash experiments/brace/run_all.sh promote-run

# Or use archive_*.sh wrappers (they call promote-run internally).
```

## Also download (for local validation)

- `experiments/brace/sync/inventories/20260801T185037Z.json`
- `experiments/brace/sync/LATEST`
- `experiments/brace/sync/CHECKLIST.md` (this file)

## Bundle summary

| Bundle | Required | Present | Missing | Git-track items |
|--------|----------|---------|---------|-----------------|
| anchor_smoke | 0 | 1 | 0 | 1 |
| confirm_archive | 4 | 0 | 4 | 6 |
| datasets_b1n1 | 3 | 0 | 3 | 5 |
| dump_pilot_archive | 3 | 2 | 2 | 6 |
| place_pilot_archive | 3 | 4 | 1 | 6 |
| replay_gate_dump | 5 | 1 | 4 | 6 |
| replay_gate_place | 5 | 0 | 5 | 6 |
| seeds_confirm | 0 | 0 | 0 | 1 |

## Download to local repo paths

Replace `REMOTE_HOST` and `REMOTE_REPO` before running.

| Status | Repo path | SHA256 | Size |
|--------|-----------|--------|------|
| present | `experiments/brace/archive/branches_place_pilot_valid_v2.3/summary.json` | `acac9d34154a626a…` | 690 |
| present | `experiments/brace/archive/branches_place_pilot_valid_v2.3/protocol.v2.3.json` | `d1692a09b49fe913…` | 2247 |
| present | `experiments/brace/archive/branches_place_pilot_valid_v2.3/analyzed_seeds.json` | `e9fcae31f87a9222…` | 242 |
| present | `experiments/brace/archive/branches_place_pilot_valid_v2.3/MANIFEST.sha256` | `df1db9017a103ff7…` | 280 |
| present | `experiments/brace/archive/branches_dump_pilot_valid_v2.3/summary.json` | `e9bc4eceb60f1223…` | 664 |
| present | `experiments/brace/archive/branches_dump_pilot_valid_v2.3/analyzed_seeds.json` | `54bd44302d339d58…` | 347 |
| present | `experiments/brace/archive/replay_audit_v2_dump_v2.3_gate/summary.json` | `0ae69b8a221bf879…` | 1969 |
| present | `experiments/brace/anchor_smoke/summary.json` | `06ca5fc4e7bfbe79…` | 659 |

## Copy commands (run on **local** machine)

```bash
mkdir -p "$(dirname experiments/brace/archive/branches_place_pilot_valid_v2.3/summary.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/branches_place_pilot_valid_v2.3/summary.json experiments/brace/archive/branches_place_pilot_valid_v2.3/summary.json
```

```bash
mkdir -p "$(dirname experiments/brace/archive/branches_place_pilot_valid_v2.3/protocol.v2.3.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/branches_place_pilot_valid_v2.3/protocol.v2.3.json experiments/brace/archive/branches_place_pilot_valid_v2.3/protocol.v2.3.json
```

```bash
mkdir -p "$(dirname experiments/brace/archive/branches_place_pilot_valid_v2.3/analyzed_seeds.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/branches_place_pilot_valid_v2.3/analyzed_seeds.json experiments/brace/archive/branches_place_pilot_valid_v2.3/analyzed_seeds.json
```

```bash
mkdir -p "$(dirname experiments/brace/archive/branches_place_pilot_valid_v2.3/MANIFEST.sha256)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/branches_place_pilot_valid_v2.3/MANIFEST.sha256 experiments/brace/archive/branches_place_pilot_valid_v2.3/MANIFEST.sha256
```

```bash
mkdir -p "$(dirname experiments/brace/archive/branches_dump_pilot_valid_v2.3/summary.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/branches_dump_pilot_valid_v2.3/summary.json experiments/brace/archive/branches_dump_pilot_valid_v2.3/summary.json
```

```bash
mkdir -p "$(dirname experiments/brace/archive/branches_dump_pilot_valid_v2.3/analyzed_seeds.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/branches_dump_pilot_valid_v2.3/analyzed_seeds.json experiments/brace/archive/branches_dump_pilot_valid_v2.3/analyzed_seeds.json
```

```bash
mkdir -p "$(dirname experiments/brace/archive/replay_audit_v2_dump_v2.3_gate/summary.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/replay_audit_v2_dump_v2.3_gate/summary.json experiments/brace/archive/replay_audit_v2_dump_v2.3_gate/summary.json
```

```bash
mkdir -p "$(dirname experiments/brace/anchor_smoke/summary.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/anchor_smoke/summary.json experiments/brace/anchor_smoke/summary.json
```

## One-shot tarball (optional, on cloud)

```bash
tar czf /tmp/brace_evidence_json.tgz \
  experiments/brace/records/ \
  experiments/brace/archive/ \
  experiments/brace/datasets/ \
  experiments/brace/seeds/place_container_plate_confirm_seeds.json \
  experiments/brace/sync/
# then: scp REMOTE_HOST:/tmp/brace_evidence_json.tgz . && tar xzf brace_evidence_json.tgz
```

## Cloud place replay gate rerun (one-time recovery)

```bash
git pull
export BRACE_TASKS=place_container_plate
export BRACE_TRACED_ROLLOUT_DIR=experiments/brace/rollouts_traced_pilot
bash experiments/brace/run_all.sh audit-v2
BRACE_PROMOTE_RUN=$(cat experiments/brace/runs/LATEST_AUDIT_place_container_plate) \
  BRACE_PROMOTE_TARGET=archive/replay_audit_v2_place_v2.3_gate \
  bash experiments/brace/run_all.sh promote-run
bash experiments/brace/run_all.sh archive-replay-gate  # optional legacy extract
bash experiments/brace/run_all.sh artifact-inventory
tar czf /tmp/brace_evidence_json.tgz experiments/brace/archive/ experiments/brace/sync/
```

## After download (local machine only)

```bash
git pull
python experiments/brace/validate_artifacts.py --inventory experiments/brace/sync/LATEST
bash experiments/brace/run_all.sh audit-mutable-paths
git add experiments/brace/archive experiments/brace/datasets experiments/brace/seeds experiments/brace/sync
git commit -m "Sync BRACE evidence JSON from cloud inventory."
git push
```

HDF5 under `experiments/brace/rollouts_traced*` is **not** tracked in git.
Manifests may reference HDF5 paths that must exist on the training host.
