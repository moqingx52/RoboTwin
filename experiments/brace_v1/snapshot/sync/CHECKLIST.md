# BRACE artifact sync checklist

## Workflow (cloud pull-only, local push)

1. **Cloud** (read-only git): `git pull` → run stages → **promote-run** → `artifact-inventory`
2. **Download** inventory + JSON files to your **local** repo (scp/rsync/tar; see below)
3. **Local**: `validate-artifacts --inventory sync/LATEST` → `git add` → `git commit` → `git push`

Cloud machines typically **cannot push**; this checklist is the handoff manifest.
Mutable working copies under `runs/` are never git-tracked.

Generated: 2026-08-01T20:09:12Z UTC
Host: gsy-ThinkStation-P3-Tower
Git commit: 7335c3e17f9ce67cf070f3f553be6891521fec08
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

- `experiments/brace/sync/inventories/20260801T200912Z.json`
- `experiments/brace/sync/LATEST`
- `experiments/brace/sync/CHECKLIST.md` (this file)

## Bundle summary

| Bundle | Required | Present | Missing | Git-track items |
|--------|----------|---------|---------|-----------------|
| anchor_smoke | 0 | 1 | 0 | 1 |
| confirm_archive | 4 | 6 | 0 | 6 |
| datasets_b1n1 | 3 | 3 | 0 | 5 |
| dump_pilot_archive | 3 | 6 | 0 | 6 |
| place_pilot_archive | 3 | 6 | 0 | 6 |
| replay_gate_dump | 5 | 3 | 3 | 6 |
| replay_gate_place | 5 | 6 | 0 | 6 |
| seeds_confirm | 0 | 1 | 0 | 1 |

## Download to local repo paths

Replace `REMOTE_HOST` and `REMOTE_REPO` before running.

| Status | Repo path | SHA256 | Size |
|--------|-----------|--------|------|
| present | `experiments/brace/archive/branches_place_pilot_valid_v2.3/checks.jsonl` | `2b5e7e2b9772125f…` | 72667 |
| present | `experiments/brace/archive/branches_place_pilot_valid_v2.3/summary.json` | `8ef8d3585c23fcde…` | 7160 |
| present | `experiments/brace/archive/branches_place_pilot_valid_v2.3/protocol.v2.3.json` | `d1692a09b49fe913…` | 2247 |
| present | `experiments/brace/archive/branches_place_pilot_valid_v2.3/source_run.json` | `d8cb7747d89a9713…` | 358 |
| present | `experiments/brace/archive/branches_place_pilot_valid_v2.3/analyzed_seeds.json` | `e9fcae31f87a9222…` | 242 |
| present | `experiments/brace/archive/branches_place_pilot_valid_v2.3/MANIFEST.sha256` | `404580f13ee4a337…` | 554 |
| present | `experiments/brace/archive/branches_dump_pilot_valid_v2.3/checks.jsonl` | `2fa31e6e45fbd9de…` | 57743 |
| present | `experiments/brace/archive/branches_dump_pilot_valid_v2.3/summary.json` | `19f0fd315df18c45…` | 6044 |
| present | `experiments/brace/archive/branches_dump_pilot_valid_v2.3/protocol.v2.3.json` | `d1692a09b49fe913…` | 2247 |
| present | `experiments/brace/archive/branches_dump_pilot_valid_v2.3/source_run.json` | `80c07d5db93ce980…` | 357 |
| present | `experiments/brace/archive/branches_dump_pilot_valid_v2.3/analyzed_seeds.json` | `54bd44302d339d58…` | 347 |
| present | `experiments/brace/archive/branches_dump_pilot_valid_v2.3/MANIFEST.sha256` | `b0ae5a804d79417d…` | 550 |
| present | `experiments/brace/archive/branches_place_confirm_v2.3/summary.json` | `fedefaadb29799aa…` | 2044 |
| present | `experiments/brace/archive/branches_place_confirm_v2.3/checks.jsonl` | `05c7262142b39d41…` | 14510 |
| present | `experiments/brace/archive/branches_place_confirm_v2.3/merged_gate.json` | `b8d109b5f41f13f2…` | 2081 |
| present | `experiments/brace/archive/branches_place_confirm_v2.3/protocol.v2.3.json` | `d1692a09b49fe913…` | 2247 |
| present | `experiments/brace/archive/branches_place_confirm_v2.3/source_run.json` | `9a3e72189b77e922…` | 354 |
| present | `experiments/brace/archive/branches_place_confirm_v2.3/confirm_seeds.json` | `c7fa8e5bab2cd963…` | 407 |
| present | `experiments/brace/datasets/place_pilot_v2.3_B1.jsonl` | `5004dabf936f6f3f…` | 8589 |
| present | `experiments/brace/datasets/place_pilot_v2.3_N1.jsonl` | `53acd528683eb524…` | 6529 |
| present | `experiments/brace/datasets/place_pilot_v2.3_summary.json` | `8b7a653b46595a61…` | 324 |
| present | `experiments/brace/archive/replay_audit_v2_place_v2.3_gate/summary.json` | `e5af1d777473c9eb…` | 3324 |
| present | `experiments/brace/archive/replay_audit_v2_place_v2.3_gate/checks.jsonl` | `29d1afeba3fb3c3a…` | 83812 |
| present | `experiments/brace/archive/replay_audit_v2_place_v2.3_gate/failures.jsonl` | `e9d395a06d47d839…` | 1039 |
| present | `experiments/brace/archive/replay_audit_v2_place_v2.3_gate/diagnostics.jsonl` | `e3b0c44298fc1c14…` | 0 |
| present | `experiments/brace/archive/replay_audit_v2_place_v2.3_gate/protocol.v2.3.json` | `d1692a09b49fe913…` | 2247 |
| present | `experiments/brace/archive/replay_audit_v2_place_v2.3_gate/source_run.json` | `b36517ad8ee96d36…` | 444 |
| present | `experiments/brace/archive/replay_audit_v2_dump_v2.3_gate/summary.json` | `0ae69b8a221bf879…` | 1969 |
| present | `experiments/brace/archive/replay_audit_v2_dump_v2.3_gate/protocol.v2.3.json` | `d1692a09b49fe913…` | 2247 |
| present | `experiments/brace/archive/replay_audit_v2_dump_v2.3_gate/source_run.json` | `c17c29454842d875…` | 384 |
| present | `experiments/brace/anchor_smoke/summary.json` | `06ca5fc4e7bfbe79…` | 659 |
| present | `experiments/brace/seeds/place_container_plate_confirm_seeds.json` | `c7fa8e5bab2cd963…` | 407 |

## Copy commands (run on **local** machine)

```bash
mkdir -p "$(dirname experiments/brace/archive/branches_place_pilot_valid_v2.3/checks.jsonl)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/branches_place_pilot_valid_v2.3/checks.jsonl experiments/brace/archive/branches_place_pilot_valid_v2.3/checks.jsonl
```

```bash
mkdir -p "$(dirname experiments/brace/archive/branches_place_pilot_valid_v2.3/summary.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/branches_place_pilot_valid_v2.3/summary.json experiments/brace/archive/branches_place_pilot_valid_v2.3/summary.json
```

```bash
mkdir -p "$(dirname experiments/brace/archive/branches_place_pilot_valid_v2.3/protocol.v2.3.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/branches_place_pilot_valid_v2.3/protocol.v2.3.json experiments/brace/archive/branches_place_pilot_valid_v2.3/protocol.v2.3.json
```

```bash
mkdir -p "$(dirname experiments/brace/archive/branches_place_pilot_valid_v2.3/source_run.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/branches_place_pilot_valid_v2.3/source_run.json experiments/brace/archive/branches_place_pilot_valid_v2.3/source_run.json
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
mkdir -p "$(dirname experiments/brace/archive/branches_dump_pilot_valid_v2.3/checks.jsonl)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/branches_dump_pilot_valid_v2.3/checks.jsonl experiments/brace/archive/branches_dump_pilot_valid_v2.3/checks.jsonl
```

```bash
mkdir -p "$(dirname experiments/brace/archive/branches_dump_pilot_valid_v2.3/summary.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/branches_dump_pilot_valid_v2.3/summary.json experiments/brace/archive/branches_dump_pilot_valid_v2.3/summary.json
```

```bash
mkdir -p "$(dirname experiments/brace/archive/branches_dump_pilot_valid_v2.3/protocol.v2.3.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/branches_dump_pilot_valid_v2.3/protocol.v2.3.json experiments/brace/archive/branches_dump_pilot_valid_v2.3/protocol.v2.3.json
```

```bash
mkdir -p "$(dirname experiments/brace/archive/branches_dump_pilot_valid_v2.3/source_run.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/branches_dump_pilot_valid_v2.3/source_run.json experiments/brace/archive/branches_dump_pilot_valid_v2.3/source_run.json
```

```bash
mkdir -p "$(dirname experiments/brace/archive/branches_dump_pilot_valid_v2.3/analyzed_seeds.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/branches_dump_pilot_valid_v2.3/analyzed_seeds.json experiments/brace/archive/branches_dump_pilot_valid_v2.3/analyzed_seeds.json
```

```bash
mkdir -p "$(dirname experiments/brace/archive/branches_dump_pilot_valid_v2.3/MANIFEST.sha256)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/branches_dump_pilot_valid_v2.3/MANIFEST.sha256 experiments/brace/archive/branches_dump_pilot_valid_v2.3/MANIFEST.sha256
```

```bash
mkdir -p "$(dirname experiments/brace/archive/branches_place_confirm_v2.3/summary.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/branches_place_confirm_v2.3/summary.json experiments/brace/archive/branches_place_confirm_v2.3/summary.json
```

```bash
mkdir -p "$(dirname experiments/brace/archive/branches_place_confirm_v2.3/checks.jsonl)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/branches_place_confirm_v2.3/checks.jsonl experiments/brace/archive/branches_place_confirm_v2.3/checks.jsonl
```

```bash
mkdir -p "$(dirname experiments/brace/archive/branches_place_confirm_v2.3/merged_gate.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/branches_place_confirm_v2.3/merged_gate.json experiments/brace/archive/branches_place_confirm_v2.3/merged_gate.json
```

```bash
mkdir -p "$(dirname experiments/brace/archive/branches_place_confirm_v2.3/protocol.v2.3.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/branches_place_confirm_v2.3/protocol.v2.3.json experiments/brace/archive/branches_place_confirm_v2.3/protocol.v2.3.json
```

```bash
mkdir -p "$(dirname experiments/brace/archive/branches_place_confirm_v2.3/source_run.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/branches_place_confirm_v2.3/source_run.json experiments/brace/archive/branches_place_confirm_v2.3/source_run.json
```

```bash
mkdir -p "$(dirname experiments/brace/archive/branches_place_confirm_v2.3/confirm_seeds.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/branches_place_confirm_v2.3/confirm_seeds.json experiments/brace/archive/branches_place_confirm_v2.3/confirm_seeds.json
```

```bash
mkdir -p "$(dirname experiments/brace/datasets/place_pilot_v2.3_B1.jsonl)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/datasets/place_pilot_v2.3_B1.jsonl experiments/brace/datasets/place_pilot_v2.3_B1.jsonl
```

```bash
mkdir -p "$(dirname experiments/brace/datasets/place_pilot_v2.3_N1.jsonl)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/datasets/place_pilot_v2.3_N1.jsonl experiments/brace/datasets/place_pilot_v2.3_N1.jsonl
```

```bash
mkdir -p "$(dirname experiments/brace/datasets/place_pilot_v2.3_summary.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/datasets/place_pilot_v2.3_summary.json experiments/brace/datasets/place_pilot_v2.3_summary.json
```

```bash
mkdir -p "$(dirname experiments/brace/archive/replay_audit_v2_place_v2.3_gate/summary.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/replay_audit_v2_place_v2.3_gate/summary.json experiments/brace/archive/replay_audit_v2_place_v2.3_gate/summary.json
```

```bash
mkdir -p "$(dirname experiments/brace/archive/replay_audit_v2_place_v2.3_gate/checks.jsonl)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/replay_audit_v2_place_v2.3_gate/checks.jsonl experiments/brace/archive/replay_audit_v2_place_v2.3_gate/checks.jsonl
```

```bash
mkdir -p "$(dirname experiments/brace/archive/replay_audit_v2_place_v2.3_gate/failures.jsonl)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/replay_audit_v2_place_v2.3_gate/failures.jsonl experiments/brace/archive/replay_audit_v2_place_v2.3_gate/failures.jsonl
```

```bash
mkdir -p "$(dirname experiments/brace/archive/replay_audit_v2_place_v2.3_gate/diagnostics.jsonl)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/replay_audit_v2_place_v2.3_gate/diagnostics.jsonl experiments/brace/archive/replay_audit_v2_place_v2.3_gate/diagnostics.jsonl
```

```bash
mkdir -p "$(dirname experiments/brace/archive/replay_audit_v2_place_v2.3_gate/protocol.v2.3.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/replay_audit_v2_place_v2.3_gate/protocol.v2.3.json experiments/brace/archive/replay_audit_v2_place_v2.3_gate/protocol.v2.3.json
```

```bash
mkdir -p "$(dirname experiments/brace/archive/replay_audit_v2_place_v2.3_gate/source_run.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/replay_audit_v2_place_v2.3_gate/source_run.json experiments/brace/archive/replay_audit_v2_place_v2.3_gate/source_run.json
```

```bash
mkdir -p "$(dirname experiments/brace/archive/replay_audit_v2_dump_v2.3_gate/summary.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/replay_audit_v2_dump_v2.3_gate/summary.json experiments/brace/archive/replay_audit_v2_dump_v2.3_gate/summary.json
```

```bash
mkdir -p "$(dirname experiments/brace/archive/replay_audit_v2_dump_v2.3_gate/protocol.v2.3.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/replay_audit_v2_dump_v2.3_gate/protocol.v2.3.json experiments/brace/archive/replay_audit_v2_dump_v2.3_gate/protocol.v2.3.json
```

```bash
mkdir -p "$(dirname experiments/brace/archive/replay_audit_v2_dump_v2.3_gate/source_run.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/replay_audit_v2_dump_v2.3_gate/source_run.json experiments/brace/archive/replay_audit_v2_dump_v2.3_gate/source_run.json
```

```bash
mkdir -p "$(dirname experiments/brace/anchor_smoke/summary.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/anchor_smoke/summary.json experiments/brace/anchor_smoke/summary.json
```

```bash
mkdir -p "$(dirname experiments/brace/seeds/place_container_plate_confirm_seeds.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/seeds/place_container_plate_confirm_seeds.json experiments/brace/seeds/place_container_plate_confirm_seeds.json
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
