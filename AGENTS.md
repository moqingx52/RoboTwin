# RoboTwin agent operating notes

These instructions apply to the whole repository. Preserve experiment evidence
and inspect active jobs before launching or stopping cloud workloads.

## Cloud GPU scheduling

The measured BRACE cloud environment is `/depot/rlinf/repos/RoboTwin` on the
host and `/workspace/RoboTwin` in Docker container `rlinf-workspace`. It has
8 NVIDIA RTX 4090 GPUs with 24 GiB each.

Use the measured capacity below unless a run has task-specific evidence that
requires a lower limit:

| Workload | Processes per assigned GPU | Measured aggregate memory per GPU |
| --- | ---: | ---: |
| Simulator collection / replay / branch / evaluation | **3** | about **15 GiB** |
| DP fine-tuning | **1** | about **22 GiB** |

Operational rules:

- Do not schedule only one collection or replay worker per GPU by default. On
  eight idle GPUs, use up to 24 workers, capped by the number of independent
  work items.
- DP fine-tuning is exclusive at one process per GPU. Do not colocate simulator
  workers on a GPU running DP fine-tuning.
- For mixed workloads, partition GPU IDs between workload classes. Apply 3
  simulator workers per simulator GPU and 1 trainer per training GPU.
- Before launching, inspect `nvidia-smi` and active processes inside the Docker
  container. Never interrupt an existing user run merely to reach the preferred
  packing ratio.
- Reduce simulator packing below 3 only for a documented OOM, a diagnostic
  isolation run, or too few independent work items. Record the reason in run
  metadata or the experiment log.

Relevant launcher settings:

```bash
# BRACE traced collection
export BRACE_GPU_IDS="0 1 2 3 4 5 6 7"
export BRACE_ROLLOUT_WORKERS_PER_GPU=3

# BRACE audit / replay / branch
export BRACE_AUDIT_WORKERS_PER_GPU=3
export BRACE_AUDIT_WORKERS=24

# Phase-1-style sharded policy evaluation / replay
export EVAL_GPU_IDS="0 1 2 3 4 5 6 7"
export EVAL_WORKERS_PER_GPU=3
export EVAL_SHARDS_PER_JOB=24
```

Most BRACE launchers already default to 3 simulator workers per GPU.
`experiments/brace/run_dump_pilot.sh` currently has a conservative default of
1, so explicitly set `BRACE_ROLLOUT_WORKERS_PER_GPU=3` when using it on this
measured 8x4090 environment. Scheduler state must still record the actual GPU
IDs, worker count, and workers-per-GPU used by each run.

## Developmental screen orchestrator

`experiments/brace/orchestrate.py screen` schedules prepare, train, and eval
jobs across the GPUs passed via `--gpus` or `BRACE_GPU_IDS` in `run_all.sh`.

Scheduling model:

- At most **one logical scheduler job per physical GPU** at a time.
- A logical eval job must use `run_eval_group.py` with **3 concurrent shards**
  on its assigned GPU. A single eval shard uses about 787 MiB in the measured
  place screen; do not serialize all 60 episodes in one process.
- DP fine-tuning remains exclusive at one process per GPU (~22 GiB). Do not
  colocate another logical job with it.
- The orchestrator must set `CUDA_VISIBLE_DEVICES` for **every** job it launches.
  `eval_per_seed.py` uses logical `cuda:0`, so without that env var all eval jobs
  pile onto physical GPU 0 and can OOM against a concurrent train job.

Recommended launch on the measured 8x4090 cloud:

```bash
export BRACE_TASKS=place_container_plate
export BRACE_GPU_IDS="0 1 2 3 4 5 6 7"
export BRACE_TRACED_ROLLOUT_DIR=experiments/brace/rollouts_traced_pilot
bash experiments/brace/run_all.sh screen
```

Resume a failed or interrupted developmental screen (after syncing code):

```bash
cd /workspace/RoboTwin
RUN_DIR=$(cat experiments/brace/runs/LATEST_SCREEN_DEVELOPMENT)

# Inspect progress before resuming
jq '{status, completed: [.jobs[]|select(.status=="completed")]|length, failed: [.jobs[]|select(.status=="failed")]|length, pending: [.jobs[]|select(.status=="pending")]|length}' \
  "${RUN_DIR}/state.json"
nvidia-smi

python experiments/brace/orchestrate.py screen \
  --resume \
  --state "${RUN_DIR}/state.json" \
  --protocol experiments/brace/screen_protocol.v1.1.json \
  --task place_container_plate \
  --run-label place_pilot_v2.3 \
  --traced-rollout-dir experiments/brace/rollouts_traced_pilot \
  --gpus 0 1 2 3 4 5 6 7 \
  --eval-workers-per-gpu 3 \
  --max-retries 1
```

Do not start a fresh `run_all.sh screen` when an existing run already has
completed jobs; `--resume` reuses the same `state.json` and keeps finished
artifacts.
