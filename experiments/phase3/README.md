# Phase 3: CPST Absorption Framework

ReGuide improves **how** recovery trajectories are generated. Phase 3 studies **how**
outcome-filtered trajectories should be **absorbed** without coverage relocation,
incorrect chunk credit, or mode collapse.

## Positioning

- **Generator (optional)**: unguided rollout or ReGuide PCG
- **Absorber (this work)**: CPST — diversity selection, verified failure-prefix,
  group-stratified replay, coverage-aware model selection

## Stages

| Stage | Status | Command |
|-------|--------|---------|
| `init` | Ready | `bash experiments/phase3/run_all.sh init` |
| `prep` | Ready | `bash experiments/phase3/run_all.sh prep` |
| `screen` | Scaffold | See finetune.sh + phase1 eval |
| `full` | Scaffold | A1 + A4 mandatory 760-ep |
| `iteration` | Scaffold | π0→π1→π2 round 2 |

## Prerequisites

1. Phase 1 rollouts at `experiments/phase1/rollouts_200/`
2. **Failure HDF5** for prefix variants — re-collect if needed:

```bash
python experiments/phase1/collect_rollouts.py \
  --task place_container_plate --task-config demo_clean \
  --seeds-file experiments/phase1/seeds/place_container_plate_seeds.json \
  --output-dir experiments/phase1/rollouts_200 \
  --save-failures --resume
```

## Prep pipeline

```bash
# Initialize state files
bash experiments/phase3/run_all.sh init

# Build A0-A4 datasets + prefix audit
bash experiments/phase3/run_all.sh prep
```

Outputs:
- `policy/DP/data_phase3/{task}-{variant}.zarr`
- `experiments/phase3/prefix_candidates/{task}.json`
- `experiments/phase3/prefix_audit/{task}.json`

## Absorption matrix (U0–U4 / A0–A4)

| ID | Data | Loss (E/S/P) | Batch budget (expert/rollout/prefix) |
|----|------|--------------|--------------------------------------|
| A0/U0 | expert only | 1/0/0 | 128/0/0 |
| A1/U1 | random success | .95/.05/0 | 122/6/0 |
| A2/U2 | diversity success | .95/.05/0 | 122/6/0 |
| A3/U3 | success + verified prefix | .95/.025/.025 | 122/3/3 |
| A4/U4 | diversity + prefix + group replay | .95/.025/.025 | 122/3/3 |

**Matched-budget rule**: U1–U4 share the same non-expert gradient mass; adding prefix
down-samples success chunks (not extra data).

## Train one variant

```bash
bash experiments/phase3/finetune.sh place_container_plate A3 0 0 5
```

## Matched-budget manifests

Training logs `loss_mass_*` via `training.log_source_loss_every`. Aggregate with
`experiments/phase3/budget.py` helpers into
`experiments/phase3/budgets/{task}_{main_id}_seed{seed}_iter{iter}.json`.

## Full eval (mandatory baselines)

After screen, **always** run 760-episode eval for **A1** and **A4** (not only the
screen winner). Use `experiments/phase1/eval_per_seed.py` with phase3 checkpoints.

## Iteration round 2 (framework)

```text
π0 → rollout D1 → CPST → π1
π1 → rollout D2 → CPST + cumulative replay → π2
```

Collect to `experiments/phase3/rollouts_iter1/`, rebuild datasets with
`--rollout-dir`, train with `finetune.sh` and `iteration` suffix (TBD in orchestrate).

## Docs

- Design: `docs/phase3_cpst_design.md`
- Phase 2b screen results: `docs/phase2b_experiment_log.md`
