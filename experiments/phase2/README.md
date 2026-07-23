# DP Data-Loop Phase 2

Phase 2 tests coverage-preserving self-training with explicit expert anchoring.
It reuses the immutable Phase 1 zarr datasets but writes independent checkpoints,
logs, evaluation results, figures, and scheduler state.

## Variants

| Variant | Dataset | Expert fraction per batch | Rollout loss weight |
|---|---|---:|---|
| `expert_only` | expert only | natural (100%) | 1 |
| `uniform_mixed` | expert + successful rollout | natural | 1 |
| `anchored_70` | expert + successful rollout | 70% | 1 |
| `anchored_50` | expert + successful rollout | 50% | 1 |
| `anchored_70_weighted` | expert + successful rollout | 70% | Phase 1 difficulty weight |

The expert fraction is enforced in every training batch by a source-aware sampler.
Difficulty weighting changes loss contribution only inside the already weighted
zarr; it does not control batch composition.

## Required Phase 1 audit

Run this before Phase 2:

```bash
AUDIT_TASKS="place_container_plate dump_bin_bigbin" \
AUDIT_TRAIN_SEEDS="0 1 2" \
bash experiments/phase1/run_all_200.sh audit 8
```

Do not proceed if the audit reports state/action misalignment, missing successful
rollouts, invalid checkpoint paths, or anomalous action/gripper statistics.

## One-command pilot

The default pilot runs both tasks, all five variants, seed 0, 50 epochs, and
learning rate `1e-5`:

```bash
bash experiments/phase2/run_all.sh
```

GPU scheduling is mutually exclusive per card:

- training mode: one DP fine-tuning process;
- evaluation mode: up to three DP evaluation processes;
- a GPU never trains and evaluates at the same time.

With eight GPUs, the scheduler initially launches up to eight training jobs.
As training jobs finish and the remaining train queue is exhausted, released GPUs
switch to three-way evaluation. This pipelines tasks across GPUs without violating
the per-card constraint.

Phase 2 opens the zarr replay buffer on disk instead of copying a private full
dataset into every training process. This prevents eight concurrent trainers from
multiplying host RAM usage. If the storage system becomes the bottleneck, cap the
number of simultaneous training GPUs; idle cards will evaluate ready checkpoints:

```bash
PHASE2_MAX_TRAIN_GPUS=2 bash experiments/phase2/run_all.sh
```

## Resume and interruption

Runtime state is atomically stored in:

```text
experiments/phase2/run_state.json
```

It records job status, PID, GPU, attempts, artifacts, group merges, timestamps,
recent events, and current GPU modes.

- Training saves numeric checkpoints every 10 epochs and resumes from the latest
  checkpoint after interruption.
- Evaluation saves every completed episode atomically and resumes missing rows.
- Completed merged evaluation groups are validated and skipped.
- `Ctrl+C` on the scheduler terminates its children, records them as resumable,
  and preserves all checkpoints.

Resume with the exact same command:

```bash
bash experiments/phase2/run_all.sh
```

The state file rejects configuration changes. Use a different
`PHASE2_STATE_PATH` for a distinct experiment.

Monitor without affecting the run:

```bash
bash experiments/phase2/watch_progress.sh
```

## Formal three-seed run

Use a separate state file:

```bash
PHASE2_TRAIN_SEEDS="0 1 2" \
PHASE2_STATE_PATH="experiments/phase2/run_state_seeds012.json" \
bash experiments/phase2/run_all.sh
```

Useful overrides:

```bash
PHASE2_EPOCHS=100
PHASE2_CHECKPOINT_EVERY=10
PHASE2_LR=5e-5
PHASE2_GPU_IDS="0 1 2 3 4 5 6 7"
PHASE2_EVAL_PER_GPU=3
PHASE2_MAX_TRAIN_GPUS=8
PHASE2_MAX_RETRIES=3
PHASE2_RETRY_BACKOFF=60
```

## Outputs

- Checkpoints: `policy/DP/checkpoints/<task>-phase2-<variant>-<seed>/`
- Logs: `experiments/phase2/logs/`
- Per-seed evaluation: `experiments/phase2/eval_results/train_seed_<seed>/`
- Aggregate: `experiments/phase2/eval_results/summary.json`
- Scheduler state: `experiments/phase2/run_state.json`
