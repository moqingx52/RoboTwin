# BRACE artifact sync checklist

Generated: 2026-08-01T18:10:28Z UTC
Host: gsy-ThinkStation-P3-Tower
Git commit: e30e150b63cc43e086fccb0baab6029e7038efb3
Protocol: v2.3

## Bundle summary

| Bundle | Required | Present | Missing | Git-track items |
|--------|----------|---------|---------|-----------------|
| anchor_smoke | 1 | 1 | 0 | 1 |
| confirm_archive | 3 | 1 | 3 | 4 |
| datasets_b1n1 | 2 | 0 | 2 | 3 |
| dump_pilot_archive | 3 | 3 | 1 | 5 |
| place_pilot_archive | 3 | 4 | 1 | 5 |
| replay_gate_dump | 1 | 1 | 0 | 1 |
| replay_gate_place | 1 | 0 | 1 | 1 |
| seeds_confirm | 0 | 0 | 0 | 1 |
| working_copy_confirm | 0 | 0 | 0 | 0 |
| working_copy_dump | 0 | 0 | 0 | 0 |
| working_copy_place | 0 | 0 | 0 | 0 |

## Download to local repo paths

Replace `REMOTE_HOST` and `REMOTE_REPO` before running.

| Status | Repo path | Source | SHA256 | Size |
|--------|-----------|--------|--------|------|
| present | `experiments/brace/archive/branches_place_pilot_valid_v2.3/summary.json` | `experiments/brace/archive/branches_place_pilot_valid_v2.3/summary.json` | `acac9d34154a626a…` | 690 |
| present | `experiments/brace/archive/branches_place_pilot_valid_v2.3/protocol.v2.3.json` | `experiments/brace/archive/branches_place_pilot_valid_v2.3/protocol.v2.3.json` | `d1692a09b49fe913…` | 2247 |
| present | `experiments/brace/archive/branches_place_pilot_valid_v2.3/analyzed_seeds.json` | `experiments/brace/archive/branches_place_pilot_valid_v2.3/analyzed_seeds.json` | `e9fcae31f87a9222…` | 242 |
| present | `experiments/brace/archive/branches_place_pilot_valid_v2.3/MANIFEST.sha256` | `experiments/brace/archive/branches_place_pilot_valid_v2.3/MANIFEST.sha256` | `df1db9017a103ff7…` | 280 |
| present | `experiments/brace/archive/branches_dump_pilot_valid_v2.3/summary.json` | `experiments/brace/archive/branches_dump_pilot_valid_v2.3/summary.json` | `e9bc4eceb60f1223…` | 664 |
| present | `experiments/brace/archive/branches_dump_pilot_valid_v2.3/protocol.v2.3.json` | `experiments/brace/protocol.v2.3.json` | `d1692a09b49fe913…` | 2247 |
| present | `experiments/brace/archive/branches_dump_pilot_valid_v2.3/analyzed_seeds.json` | `experiments/brace/archive/branches_dump_pilot_valid_v2.3/analyzed_seeds.json` | `54bd44302d339d58…` | 347 |
| present | `experiments/brace/archive/branches_place_confirm_v2.3/protocol.v2.3.json` | `experiments/brace/protocol.v2.3.json` | `d1692a09b49fe913…` | 2247 |
| present | `experiments/brace/archive/replay_audit_v2_dump_v2.3_gate/summary.json` | `experiments/brace/archive/replay_audit_v2_dump_v2.3_gate/summary.json` | `0ae69b8a221bf879…` | 1969 |
| present | `experiments/brace/anchor_smoke/summary.json` | `experiments/brace/anchor_smoke/summary.json` | `06ca5fc4e7bfbe79…` | 659 |

## Copy commands

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
mkdir -p "$(dirname experiments/brace/archive/branches_dump_pilot_valid_v2.3/protocol.v2.3.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/protocol.v2.3.json experiments/brace/archive/branches_dump_pilot_valid_v2.3/protocol.v2.3.json
```

```bash
mkdir -p "$(dirname experiments/brace/archive/branches_dump_pilot_valid_v2.3/analyzed_seeds.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/branches_dump_pilot_valid_v2.3/analyzed_seeds.json experiments/brace/archive/branches_dump_pilot_valid_v2.3/analyzed_seeds.json
```

```bash
mkdir -p "$(dirname experiments/brace/archive/branches_place_confirm_v2.3/protocol.v2.3.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/protocol.v2.3.json experiments/brace/archive/branches_place_confirm_v2.3/protocol.v2.3.json
```

```bash
mkdir -p "$(dirname experiments/brace/archive/replay_audit_v2_dump_v2.3_gate/summary.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/archive/replay_audit_v2_dump_v2.3_gate/summary.json experiments/brace/archive/replay_audit_v2_dump_v2.3_gate/summary.json
```

```bash
mkdir -p "$(dirname experiments/brace/anchor_smoke/summary.json)"
scp REMOTE_HOST:/workspace/RoboTwin/experiments/brace/anchor_smoke/summary.json experiments/brace/anchor_smoke/summary.json
```

## After download (local machine)

```bash
bash experiments/brace/archive_place_pilot.sh
bash experiments/brace/archive_dump_pilot.sh
bash experiments/brace/archive_place_confirm.sh  # if confirm artifacts present
python experiments/brace/validate_artifacts.py --inventory experiments/brace/sync/LATEST
```

HDF5 under `experiments/brace/rollouts_traced*` is **not** tracked in git.
Manifests may reference HDF5 paths that must exist on the training host.
