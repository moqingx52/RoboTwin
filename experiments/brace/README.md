# BRACE experiment scaffold

This directory is reserved for the revised B1--B3 experiments.

Authoritative design and cloud gates:

```text
docs/brace_cloud_experiment_plan.md
```

Implementation order:

1. `replay_audit.py` — deterministic prefix reconstruction and tolerance audit
2. `collect_branches.py` — candidate/control chunks with matched continuation seeds
3. `aggregate_advantage.py` — bootstrap LCB and accepted-chunk manifest
4. frozen-teacher denoiser constraint in the DP workspace
5. `orchestrate.py` — resumable smoke, screen, full, and iteration stages

Until these files exist and pass smoke tests, do not mark BRACE results as completed in
`docs/paper/sections/06_results.tex`.

The existing `experiments/phase3/` directory supplies U0--U4 baselines.

## Unified cloud entry

The current failure-HDF5 collection is useful input for BRACE. Let it finish, but do not
continue into the old Phase 3 `prep/screen/full` stages.

```bash
# Read-only progress; safe while the existing launcher is running.
bash experiments/brace/run_all.sh status

# After all collection workers finish. The old launcher normally verifies and merges
# automatically; this command is an idempotent strict re-check.
BRACE_NUM_SHARDS=12 \
bash experiments/brace/run_all.sh verify

# Create BRACE directories and protocol.json.
bash experiments/brace/run_all.sh init
```

Before `audit`, replace every `FREEZE_BEFORE_RUN` value in
`experiments/brace/protocol.json` and change its status to `frozen`.

The replay gate requires `/joint_action/vector`, both `/endpose/*_endpose`
datasets, and `/task_object_pose/<name>` in every sampled HDF5. The latter is
recorded by the current `demo_clean` config for the two BRACE tasks. Rollouts
collected before that field was added must be recollected; the audit deliberately
reports them as an incomplete no-go instead of falling back to a robot-only check.

```bash
bash experiments/brace/run_all.sh audit
bash experiments/brace/run_all.sh branch
bash experiments/brace/run_all.sh screen
bash experiments/brace/run_all.sh full
```

The entry intentionally refuses to advance when a required implementation or gate artifact
is missing. It never delegates to the old CPST `prep/screen/full` stages.
