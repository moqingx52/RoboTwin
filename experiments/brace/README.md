# BRACE experiment scaffold

This directory is reserved for the revised B1--B3 experiments.

Authoritative design:

```text
docs/brace_cloud_experiment_plan.md
docs/brace_audit_v2_design.md
```

## Stage 1 gates

| Version | Script | Protocol | Purpose |
|---------|--------|----------|---------|
| v1 (archived) | `replay_audit.py` | `archive/protocol.v1.frozen.json` | Waypoint replay baseline (80% NO-GO) |
| **v2.3 (current)** | `replay_audit_v2.py` | `protocol.v2.3.json` | Chunk-boundary branch harness + symmetry-aware actor metrics |
| v2.2 (archived) | `replay_audit_v2.py` | `protocol.v2.2.json` | Symmetry-aware actor metrics + separate gates |
| v2.1 (archived) | `replay_audit_v2.py` | `protocol.v2.1.json` | Separate restore + replay gates |
| v2.0 (diagnostic) | `replay_audit_v2.py` | `protocol.v2.json` | Mixed pass-rate gate (superseded) |

v1 results are archived under `experiments/brace/archive/replay_audit_v1_no_go/`.
The first v2.0 mixed-gate run is documented under
`experiments/brace/archive/replay_audit_v2_mixed_gate_diagnostic/`.
The v2.1 gate run is archived under
`experiments/brace/archive/replay_audit_v2_v2.1_gate/`.

Stage 2 branch pilots:

- **Invalid** (v2.2 harness): `archive/branches_place_pilot_invalid_v2.2/`
- **Valid place** (v2.3, GO +62.2 pp): `archive/branches_place_pilot_valid_v2.3/`
- **Valid dump** (v2.3, GO +36.1 pp): `archive/branches_dump_pilot_valid_v2.3/` · `branches_dump/`
- Conclusion write-ups:
  - Place: `docs/brace_stage2_place_pilot_conclusion.md`
  - Dump: `docs/brace_stage2_dump_pilot_conclusion_20260801.md`
  - Rolling log: `docs/brace_experiment_log.md`
  - Roadmap: `docs/brace_track_abc_roadmap.md`

`branch` requires each task in `BRACE_TASKS` to pass replay audit v2
(`tasks.<task>.replay_gate_passed=true`), resolved via `runs/LATEST_AUDIT_<task>`
→ `archive/replay_audit_v2_*_gate/` → legacy path (deprecated).
For a single-task pilot, set `BRACE_TASKS=place_container_plate` and use
`experiments/brace/rollouts_traced_pilot/` via `run_place_pilot.sh`.
For dump A2: `bash experiments/brace/run_dump_pilot.sh` → `runs/.../branches_dump/`.

## Unified cloud entry

```bash
# Legacy rollouts (policy eval / training; not branch-ready)
bash experiments/brace/run_all.sh status
bash experiments/brace/run_all.sh verify

# v2 traced rollouts (schema v2 HDF5 under rollouts_traced/)
bash experiments/brace/run_all.sh collect-trace-smoke   # 2-4 per task
bash experiments/brace/run_all.sh collect-trace-audit   # 20 per task
bash experiments/brace/run_all.sh verify-traced

# Stage 2 place pilot (always pin protocol v2.3 explicitly)
BRACE_PROTOCOL_V2_PATH=experiments/brace/protocol.v2.3.json \
BRACE_TASKS=place_container_plate \
BRACE_TRACED_ROLLOUT_DIR=experiments/brace/rollouts_traced_pilot \
  bash experiments/brace/run_all.sh audit-v2

BRACE_PROTOCOL_V2_PATH=experiments/brace/protocol.v2.3.json \
BRACE_TASKS=place_container_plate \
BRACE_TRACED_ROLLOUT_DIR=experiments/brace/rollouts_traced_pilot \
  bash experiments/brace/run_all.sh branch

# After a passing screen.v1.1 executable anchor smoke:
BRACE_TASKS=place_container_plate bash experiments/brace/run_all.sh anchor-smoke
BRACE_TASKS=place_container_plate bash experiments/brace/run_all.sh screen
bash experiments/brace/run_all.sh full
```

Traced collection uses `task_config/demo_brace_trace.yml` and extends HDF5 with:

- `/policy_chunks`, `/control_trace`, `/branch_snapshots`
- all dynamic actors (`deskbin` + `garbage_*` for `dump_bin_bigbin`)

The entry never delegates to the old CPST `prep/screen/full` stages.

## Evidence sync (cloud pull-only → local push)

### Data management policy (immutable runs)

Shared paths like `replay_audit_v2/summary.json` and `branches/` are **legacy and
overwrite-prone**. Default behavior (since 2026-08-02):

```text
experiments/brace/runs/<UTC>_<stage>_<task>/meta.json
experiments/brace/runs/<UTC>_<stage>_<task>/replay_audit_v2/<task>/summary.json
experiments/brace/runs/<UTC>_branch_<label>/branches/summary.json
```

Pointers (cloud-local, not git-tracked):

- `runs/LATEST` — most recent run directory
- `runs/LATEST_AUDIT_<task>` — latest audit output for a task
- `runs/LATEST_<branch_label>` — latest branch output
- `records/<UTC>_<stage>_<task>.json` — stage metadata (auto-emitted)
- `records/index.jsonl` — chronological index; `bash experiments/brace/run_all.sh list-records`
- `records/bundles/<record_id>/` — copied JSON/JSONL evidence (no HDF5/checkpoints)

Human conclusions belong in `docs/` (private). Machine gate metadata and a
self-contained JSON evidence copy go to `records/`. Full artifacts stay in
`runs/`; git-synced evidence only after `promote-run` → `archive/`.

**Promote** validated outputs into frozen `archive/` before syncing to git:

`audit-v2`, `export-verified-chunks`, and the place/dump/confirm wrapper scripts
promote automatically after a passing stage. Set `BRACE_AUTO_PROMOTE=0` for a
recovery rerun or when selecting a new versioned target. Failed stages are not
promoted, but their JSON evidence remains in `records/bundles/`.

```bash
# After audit-v2
BRACE_PROMOTE_RUN=$(cat experiments/brace/runs/LATEST_AUDIT_place_container_plate) \
  BRACE_PROMOTE_TARGET=archive/replay_audit_v2_place_v2.3_gate \
  bash experiments/brace/run_all.sh promote-run

# After branch (or use archive_place_pilot.sh wrapper)
BRACE_PROMOTE_RUN=$(cat experiments/brace/runs/LATEST_branches) \
  BRACE_PROMOTE_TARGET=archive/branches_place_pilot_valid_v2.3 \
  bash experiments/brace/run_all.sh promote-run

# After export-verified-chunks
BRACE_PROMOTE_RUN=$(cat experiments/brace/runs/LATEST_export_place_pilot_v2.3) \
  BRACE_PROMOTE_TARGET=datasets/ \
  bash experiments/brace/run_all.sh promote-run
```

Set `BRACE_LEGACY_MUTABLE_OUTPUTS=1` only to reproduce old scripts.
Run `bash experiments/brace/run_all.sh audit-mutable-paths` in CI to catch regressions.

```bash
bash experiments/brace/run_all.sh list-runs
```

Cloud machines usually **can pull but not push**. Evidence JSON flows:

```text
Cloud: git pull → stages → promote-run → artifact-inventory
          ↓ scp/rsync/tar (CHECKLIST.md lists paths + SHA256)
Local: validate-artifacts → audit-mutable-paths → git add → git commit → git push
```

**On cloud** (scan only, no git commit):

```bash
git pull
bash experiments/brace/run_all.sh artifact-inventory
# → experiments/brace/sync/CHECKLIST.md
# → experiments/brace/sync/inventories/<timestamp>.json
```

**On local** (after downloading files per CHECKLIST):

```bash
git pull
python experiments/brace/validate_artifacts.py --inventory experiments/brace/sync/LATEST
git add experiments/brace/archive experiments/brace/datasets experiments/brace/seeds experiments/brace/sync
git commit -m "Sync BRACE evidence JSON from cloud."
git push
```

Optional one-shot from cloud: tarball `archive/`, `datasets/`, `sync/` and scp to local.

Per-task replay audit outputs should live under `replay_audit_v2/<task>/`;
extract gate JSON with `bash experiments/brace/run_all.sh archive-replay-gate` before inventory.
HDF5 under `rollouts_traced*` is not tracked in git.

## Developmental place screen

`screen_protocol.v1.json` remains the original frozen design. The executable
optimizer/data settings are versioned in `screen_protocol.v1.1.json`; existing v1
evidence is never rewritten. The trainer now provides:

- matched B1/N1 chunk zarr construction without writing to `rollouts_traced_pilot/`;
- B1/N1 SFT and B2/B3 frozen raw-teacher constraints with shared action/noise/timestep;
- checkpoint/resume with stable teacher hash, dual variables, LR and EMA schedule position;
- fixed 20 ID + 20 train + 20 Hard evaluation at epochs 1/3/5/7/10;
- timestamped run state plus automatic JSON evidence bundling in `records/bundles/`.

U1 aliases the matched-random N1 data/trajectory in this developmental screen, so
U1 and B2 have identical optimizer examples. The 11+11 manifests are explicitly
developmental evidence and cannot produce a paper-ready promotion.
