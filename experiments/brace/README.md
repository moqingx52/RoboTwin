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
| **v2 (current)** | `replay_audit_v2.py` | `protocol.v2.json` | Snapshot restore + exact control-trace replay |

v1 results are archived under `experiments/brace/archive/replay_audit_v1_no_go/`.
Do not overwrite them. `branch` requires **v2** `replay_audit_v2/summary.json` with `passed=true`.

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
