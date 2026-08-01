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
- **Valid** (v2.3 harness, place GO): `archive/branches_place_pilot_valid_v2.3/`
- Conclusion write-up: `docs/brace_stage2_place_pilot_conclusion.md`

`branch` requires each task in `BRACE_TASKS` to have
`replay_audit_v2/summary.json` → `tasks.<task>.replay_gate_passed=true`.
For a single-task pilot, set `BRACE_TASKS=place_container_plate` and use
`experiments/brace/rollouts_traced_pilot/` via `run_place_pilot.sh`.

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
