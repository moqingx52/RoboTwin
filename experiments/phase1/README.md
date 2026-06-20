# DP Phase1 Diagnostics

This folder implements the first-stage RoboTwin DP experiment without RLinf.

## Tasks

- `move_can_pot`
- `place_container_plate`
- `click_alarmclock`
- `dump_bin_bigbin`

All commands below use `demo_clean` because the official DP Easy baseline is in the useful success-rate range for success-filtered self-training.

## End-to-End

Run one task:

```bash
bash experiments/phase1/run_task.sh click_alarmclock 0 all
```

Run all four tasks on an 8-GPU node:

```bash
bash experiments/phase1/run_all.sh all 8
```

The `rollout` stage uses 8 stable independent workers by default:

```text
4 tasks x 2 seed shards per task = 8 processes
```

Each worker writes its own `manifest_shard_XX_of_02.jsonl`; after all workers finish,
`run_all.sh` merges them into `manifest.jsonl` and `seed_stats.json`.

Run a single stage:

```bash
bash experiments/phase1/run_task.sh move_can_pot 0 base
bash experiments/phase1/run_task.sh move_can_pot 0 seeds
bash experiments/phase1/run_task.sh move_can_pot 0 rollout
bash experiments/phase1/run_task.sh move_can_pot 0 rollout 0 2
bash experiments/phase1/run_task.sh move_can_pot 1 rollout 1 2
bash experiments/phase1/run_task.sh move_can_pot 0 merge_rollout
bash experiments/phase1/run_task.sh move_can_pot 0 build
bash experiments/phase1/run_task.sh move_can_pot 0 finetune
bash experiments/phase1/run_task.sh move_can_pot 0 eval
```

## Stage Outputs

- Seed splits: `experiments/phase1/seeds/{task}_seeds.json`
- Rollout manifest: `experiments/phase1/rollouts/{task}/manifest.jsonl`
- Rollout shard manifests: `experiments/phase1/rollouts/{task}/manifest_shard_XX_of_YY.jsonl`
- Per-seed base success rates: `experiments/phase1/rollouts/{task}/seed_stats.json`
- Variant zarr datasets:
  - `policy/DP/data/{task}-success.zarr`
  - `policy/DP/data/{task}-seed_balanced.zarr`
  - `policy/DP/data/{task}-difficulty_weighted.zarr`
- Evaluation JSON: `experiments/phase1/eval_results/{task}/{variant}.json`
- Figures: `experiments/phase1/figures/{task}/`

## Variants

- `base`: original DP trained from 50 expert demonstrations.
- `success`: expert data plus all successful DP-Base rollouts.
- `seed_balanced`: expert data plus at most one successful rollout per environment seed.
- `difficulty_weighted`: same episodes as `success`, but rollout chunks are weighted by
  `clip(1 / (J_hat + 0.05) ** 0.5, 1, 4)`.

## Dry Runs

Use dry runs to verify file paths and JSON formats without launching simulation:

```bash
python experiments/phase1/generate_seed_splits.py --task click_alarmclock --dry-run
python experiments/phase1/collect_rollouts.py --task click_alarmclock --dry-run
python experiments/phase1/eval_per_seed.py --task click_alarmclock --variant base --ckpt-path policy/DP/checkpoints/dummy.ckpt --dry-run
```

