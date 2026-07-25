# DP Data-Loop Phase 2b

Phase 2b isolates collapse causes before spending GPU budget on more seeds.
It adds:

- frozen Base normalizer during mixed fine-tuning;
- source-separated expert/rollout loss;
- epoch 5/10/20 preservation screens;
- normalizer swap diagnose without training.

Do **not** rerun `experiments/phase2/run_all.sh` while Phase 2b is active.

## Stages

| Stage | Purpose |
|---|---|
| `diagnose` | A0/A1/A2 normalizer swap eval + expert/rollout stats |
| `screen` | B0–B4 short training with epoch 5/10/20 gates |
| `full_seed0` | Full 760-episode eval for top 1–2 screen winners |
| `confirm_seeds` | Train/eval seeds 1 and 2 for the single full winner |

## Prerequisites

```bash
AUDIT_TASKS="place_container_plate dump_bin_bigbin" \
AUDIT_TRAIN_SEEDS="0 1 2" \
bash experiments/phase1/run_all_200.sh audit 8
```

Required artifacts:

- `policy/DP/checkpoints/{task}-demo_clean-200-0/600.ckpt`
- `policy/DP/data_phase1_200/{task}-expert_only.zarr`
- `policy/DP/data_phase1_200/{task}-success.zarr`
- `experiments/phase1/eval_results_200/train_seed_0/{task}/base.json`

## GPU layout (default)

- GPUs `0–6`: work GPUs (`PHASE2B_GPU_IDS`)
- GPU `7`: kept idle
- Per GPU: **1 train** or up to **3 eval** shards, never both

## Commands

### Stage A — diagnose

```bash
PHASE2B_GPU_IDS="0 1 2 3 4 5 6" \
PHASE2B_STATE_PATH="experiments/phase2b/run_state_diagnose.json" \
bash experiments/phase2b/run_all.sh diagnose
```

Monitor:

```bash
bash experiments/phase2b/watch_progress.sh experiments/phase2b/run_state_diagnose.json
```

Outputs:

- `experiments/phase2b/diagnose/normalizer_ckpts.json`
- `experiments/phase2b/diagnose/source_stats.json`
- `experiments/phase2b/diagnose/eval/{task}/A{0,1,2}_*.json`
- `experiments/phase2b/diagnose/diagnose_summary.json`

Decision: if `A2` ID mean SR `< 80%` of `A0`, normalizer replacement is the primary suspect.

### Stage B+C — screen

```bash
PHASE2B_GPU_IDS="0 1 2 3 4 5 6" \
PHASE2B_MAX_TRAIN_GPUS=7 \
PHASE2B_STATE_PATH="experiments/phase2b/run_state_screen.json" \
bash experiments/phase2b/run_all.sh screen
```

Candidates:

| ID | Batch E/R | Loss |
|---|---|---|
| B0 | 100/0 | expert only, frozen normalizer |
| B1 | 90/10 | batch mean |
| B2 | 90/10 | 0.9·LE + 0.1·LR |
| B3 | 95/5 | 0.95·LE + 0.05·LR |
| B4 | 90/10 | 0.95·LE + 0.05·LR |

Screen eval per checkpoint: 20 ID × 1 + 10 train × 1 = 30 episodes.

Promotion uses Base-relative thresholds (ID ≥ 80%, train ≥ 75%, no >10pp drop vs previous checkpoint).

### Stage D — full seed-0

```bash
PHASE2B_GPU_IDS="0 1 2 3 4 5 6" \
PHASE2B_STATE_PATH="experiments/phase2b/run_state_full_seed0.json" \
PHASE2B_SCREEN_STATE_PATH="experiments/phase2b/run_state_screen.json" \
bash experiments/phase2b/run_all.sh full_seed0
```

Only the best 1–2 `(candidate, epoch)` pairs from screen are evaluated at full protocol (760 episodes).

### Stage E — confirm seeds

Run only when exactly one candidate passes full seed-0 gates:

```bash
PHASE2B_GPU_IDS="0 1 2 3 4 5 6" \
PHASE2B_STATE_PATH="experiments/phase2b/run_state_confirm.json" \
PHASE2B_FULL_STATE_PATH="experiments/phase2b/run_state_full_seed0.json" \
bash experiments/phase2b/run_all.sh confirm_seeds
```

## Resume / retry

State is atomic JSON per stage. Resume with the same command.

Retry failed jobs:

```bash
PHASE2B_RETRY_FAILED=1 bash experiments/phase2b/run_all.sh screen
```

## Training changes (DP)

New Hydra fields under `training`:

- `normalizer_source`: `dataset` (default, Phase 2 behavior) or `checkpoint` (Phase 2b default via `finetune.sh`)
- `loss_mode`: `pooled` or `source_separated`
- `lambda_expert`, `lambda_rollout`
- `log_source_loss_every`, `log_source_grad_norm_every`

Batches now carry `sample_source` (0=expert, 1=rollout) for logging and source-separated loss.

## Outputs

- Checkpoints: `policy/DP/checkpoints/{task}-phase2b-{candidate}-{seed}/`
- Logs: `experiments/phase2b/logs/`
- Screen eval: `experiments/phase2b/eval_results/train_seed_0/screen/`
- Full eval: `experiments/phase2b/eval_results/train_seed_{seed}/`
- Promotion records: `experiments/phase2b/promotions/`
