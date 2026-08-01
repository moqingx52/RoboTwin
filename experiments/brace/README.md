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

`branch` requires each task in `BRACE_TASKS` to have
`replay_audit_v2/summary.json` → `tasks.<task>.replay_gate_passed=true`.
For a single-task pilot, set `BRACE_TASKS=place_container_plate` and use
`experiments/brace/rollouts_traced_pilot/` via `run_place_pilot.sh`.
For dump A2: `bash experiments/brace/run_dump_pilot.sh` → `branches_dump/`.

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

bash experiments/brace/run_all.sh screen
bash experiments/brace/run_all.sh full
```

Traced collection uses `task_config/demo_brace_trace.yml` and extends HDF5 with:

- `/policy_chunks`, `/control_trace`, `/branch_snapshots`
- all dynamic actors (`deskbin` + `garbage_*` for `dump_bin_bigbin`)

The entry never delegates to the old CPST `prep/screen/full` stages.

## Evidence sync (cloud without git push)

Remote runs can scan artifacts and emit a download checklist:

```bash
bash experiments/brace/run_all.sh artifact-inventory
# → experiments/brace/sync/inventories/<timestamp>.json
# → experiments/brace/sync/CHECKLIST.md
```

After copying JSON files to the local repo paths listed in the checklist:

```bash
bash experiments/brace/archive_place_pilot.sh
bash experiments/brace/archive_dump_pilot.sh
bash experiments/brace/archive_place_confirm.sh
python experiments/brace/validate_artifacts.py --inventory experiments/brace/sync/LATEST
```

Per-task replay audit outputs should live under `replay_audit_v2/<task>/` and be
merged with `bash experiments/brace/run_all.sh merge-audit-v2`.
HDF5 under `rollouts_traced*` is not tracked in git.
