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
| **v2.1 (current)** | `replay_audit_v2.py` | `protocol.v2.1.json` | Separate restore + replay gates |
| v2.0 (diagnostic) | `replay_audit_v2.py` | `protocol.v2.json` | Mixed pass-rate gate (superseded) |

v1 results are archived under `experiments/brace/archive/replay_audit_v1_no_go/`.
The first v2.0 mixed-gate run is documented under
`experiments/brace/archive/replay_audit_v2_mixed_gate_diagnostic/`.

`branch` requires each task in `BRACE_TASKS` to have
`replay_audit_v2/summary.json` → `tasks.<task>.replay_gate_passed=true`.
For a single-task pilot, set `BRACE_TASKS=place_container_plate` after re-running
`audit-v2` under protocol v2.1.

## Unified cloud entry

```bash
# Legacy rollouts (policy eval / training; not branch-ready)
bash experiments/brace/run_all.sh status
bash experiments/brace/run_all.sh verify

# v2 traced rollouts (schema v2 HDF5 under rollouts_traced/)
bash experiments/brace/run_all.sh collect-trace-smoke   # 2-4 per task
bash experiments/brace/run_all.sh collect-trace-audit   # 20 per task
bash experiments/brace/run_all.sh verify-traced
bash experiments/brace/run_all.sh audit-v2

# After v2 gate passes
bash experiments/brace/run_all.sh branch
bash experiments/brace/run_all.sh screen
bash experiments/brace/run_all.sh full
```

Traced collection uses `task_config/demo_brace_trace.yml` and extends HDF5 with:

- `/policy_chunks`, `/control_trace`, `/branch_snapshots`
- all dynamic actors (`deskbin` + `garbage_*` for `dump_bin_bigbin`)

The entry never delegates to the old CPST `prep/screen/full` stages.
