# RoboTwin agent operating notes

These instructions apply to the whole repository. Preserve experiment evidence
and inspect active jobs before launching or stopping cloud workloads.

## Paper north star: do not let infrastructure become the research question

The BRACE paper studies whether a policy can improve from its own experience
when useful self-generated chunks are branch/replay verified, while an anchor
preserves capabilities the base policy already had. The primary comparisons are
verified versus matched-random self-generated data, with and without the anchor.

Expert demonstration collection is only bootstrap infrastructure for the base
DP policy. Do not turn expert-seed determinism, repeated expert-script probing,
or individual simulator flakes into the main project. A diagnostic may explain
an artifact, but it must not block independent collection, training, Line A
method development, or other tasks.

Operational failure and scientific validity are different:

- A failed task/job must not stop unrelated scheduler jobs. Continue other
  tasks and record the failed task explicitly.
- A failed gate must not be silently ignored in a paper claim. Report missing or
  infeasible tasks and apply the frozen protocol's replacement, denominator, or
  sensitivity rule.
- Never delete a task, replace a seed, change a denominator, or tune a method
  using held-out learned-policy outcomes without an explicit new protocol
  revision.

The current cloud handover/collection-path diagnostics are supporting evidence
only. Regardless of whether they pass, archive their JSON and continue the main
pipeline. Do not require further expert stability diagnostics unless a task
cannot produce the target number of successful demonstrations within its fixed
acquisition budget.

## Base dataset policy for the paper

Use **200 successful `demo_clean` expert trajectories per task** for the primary
BRACE paper experiments. This matches the substrate used by the established
BRACE development/confirmatory pipeline and the project's empirical finding
that DP generally needs about 200 demonstrations for mean success above 50%.

Keep **50 demonstrations** only as the RoboTwin official-alignment baseline or
data-scale sensitivity. Do not make the 50-demo checkpoint the primary BRACE
self-improvement substrate merely because it is the official example setting.
All causal BRACE arms for a task must start from the same frozen Base200
checkpoint and dataset.

Collect expert data with the native success-first RoboTwin semantics:

1. Traverse a preregistered, task-specific candidate seed sequence.
2. Attempt each seed normally; on success, immediately save the seed and the
   trajectory together.
3. Log failures and continue to the next seed until 200 successful trajectories
   are materialized or the fixed acquisition budget is exhausted.
4. Freeze the materialized unit: seed + trajectory + task config + code commit +
   attempt manifest. Do not freeze a seed first and later require the planner to
   reproduce it.
5. Never consult learned-policy performance when selecting expert trajectories.

Repeated success of the same expert seed is not a paper gate. Bounded same-seed
retry may be used only when frozen in an operational amendment, but ordinary
success-first collection should be preferred. A task that exhausts its fixed
budget is marked expert-acquisition-infeasible; it does not halt other tasks.

The frozen 50-demo multitask v1 design and its partial checkpoints remain pilot
evidence. Do not rewrite its files in place. The 200-demo primary experiment
must use a new versioned protocol and artifact names (`multitask v2`,
`*-demo_clean-200-*`).

## Main critical path

Prioritize work in this order:

1. Freeze the BRACE-v2 intervention method on development task
   `place_container_plate` (Line A); never tune it on held-out tasks.
2. Materialize Base200 expert datasets and train Base200 DP checkpoints across
   tasks; task jobs continue independently on failure.
3. Collect Base200 policy rollouts with successes, failures, control traces, and
   branch snapshots.
4. Replay/branch verify candidate self-generated chunks and construct matched
   random controls.
5. Train U0/N1/B1/B2/B3 with matched optimizer steps and data budgets.
6. Run frozen preservation/adaptation evaluation and paired task-level
   inference.

Before doing work outside this list, state which paper claim it unblocks. If it
does not unblock a claim, a required artifact, safety, or reproducibility, defer
it. In particular, do not start another broad seed-feasibility campaign, expert
script repair program, or simulator determinism study merely to make all setup
artifacts look perfect.

The detailed execution plan is `docs/brace_paper_execution_plan_v2.md`.

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
export BRACE_TRACED_ROLLOUT_DIR=experiments/brace/rollouts_traced
bash experiments/brace/run_all.sh screen
```

Anchor smoke, anchor-feasibility, and screen require the full traced corpus
(`rollouts_traced`) so anchor replay can populate `base_solved` and `boundary`.
Use `rollouts_traced_pilot` only for Stage-2 branch collection.

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
  --traced-rollout-dir experiments/brace/rollouts_traced \
  --gpus 0 1 2 3 4 5 6 7 \
  --eval-workers-per-gpu 3 \
  --max-retries 1
```

Do not start a fresh `run_all.sh screen` when an existing run already has
completed jobs; `--resume` reuses the same `state.json` and keeps finished
artifacts.
