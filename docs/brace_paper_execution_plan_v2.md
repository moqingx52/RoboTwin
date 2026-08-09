# BRACE paper execution plan v2

Date: 2026-08-09  
Status: planning document; freeze machine-readable protocol before confirmatory held-out training.

## 1. Research question

The paper asks whether a robot policy can improve from its own rollout data when
useful chunks are causally verified, without catastrophically forgetting the
capabilities already present in the base policy.

The primary scientific claims are:

1. **Credit:** branch-verified self-generated chunks improve adaptation more
   than budget-matched random successful chunks.
2. **Preservation:** a frozen base-policy anchor reduces loss of previously
   solved behavior.
3. **Pareto improvement:** combining verified chunks and the anchor improves the
   preservation/adaptation trade-off across held-out tasks.

Expert collection, checkpoint conversion, replay tooling, and seed management
support these claims; they are not themselves the main contribution.

## 2. Versioning decision

Preserve `brace.multitask.v1` as the 50-demo official-alignment pilot. Its
partial datasets/checkpoints and feasibility diagnostics remain immutable pilot
evidence and must not be relabeled as the final experiment.

Create `brace.multitask.v2` for the primary paper experiment:

- Base policy: DP trained from 200 successful `demo_clean` trajectories.
- Official sensitivity: Base50, reported separately and not used to initialize
  primary BRACE arms.
- All U0/N1/B1/B2/B3 arms for a task start from the identical Base200 checkpoint.
- Existing `*-demo_clean-200-0/600.ckpt` may be reused only when dataset and
  checkpoint provenance match the v2 acquisition manifest.

The reason for Base200 must be recorded as a pre-study design decision: prior
project evidence found that roughly 200 demonstrations are needed for DP mean
success above 50%, while the earlier BRACE pipeline already used Base200. Do not
tune the demonstration count per held-out task.

## 3. Task panel and non-blocking feasibility

Use `place_container_plate` only for method development. Do not use held-out
policy results to choose the method, hyperparameters, anchor weight, checkpoint,
or task panel.

Before Base policy evaluation, preregister:

- ten primary held-out tasks;
- an ordered reserve list of two or more tasks selected from public benchmark
  context and skill-family coverage only;
- a fixed expert-acquisition budget per task;
- the replacement rule for expert-acquisition-infeasible tasks.

Recommended acquisition budget: traverse at most 5,000 preregistered candidate
seeds to obtain 200 successful trajectories. Use a task-specific seed namespace
that is disjoint from all train-rollout, census, confirm-easy, confirm-hard, and
policy RNG partitions.

If a task yields fewer than 200 demonstrations within budget:

1. record `expert_acquisition_infeasible` and preserve its full attempt manifest;
2. continue all other jobs;
3. before any held-out policy evaluation, apply the frozen reserve-task rule;
4. retain the infeasible task in the feasibility appendix;
5. never substitute a task after viewing BRACE or Base policy performance.

This is operationally fail-open and scientifically explicit: one task cannot
stall the cluster, but the final paper cannot silently change its denominator.

## 4. Base200 acquisition

### 4.1 Collection rule

For every task, consume candidate seeds in fixed manifest order. Each seed gets
one normal RoboTwin expert attempt. A successful attempt immediately writes:

- HDF5 trajectory;
- seed and episode index;
- task config SHA256;
- code commit;
- success/check result;
- task-specific acquisition manifest row.

Failed attempts are logged and skipped. Continue until 200 successes or the
5,000-seed budget is exhausted. The saved successful trajectories, rather than
the ability to regenerate their seeds, are the frozen dataset.

Do not use learned-policy outcomes, future evaluation seeds, or task-specific
success-rate targets to choose demonstrations.

### 4.2 Scheduling

- Three simulator collection processes per assigned 4090.
- Up to 24 concurrent processes on eight idle GPUs, capped by independent task
  jobs/shards.
- A failed task process records failure and does not terminate unrelated jobs.
- Archive JSON/JSONL metadata; keep HDF5/zarr outside git with immutable run
  provenance.

### 4.3 Dataset acceptance

A Base200 dataset is ready when it has exactly 200 successful HDF5 episodes,
unique recorded episode indices, a complete attempt manifest, valid config/code
provenance, and a successfully processed zarr. Expert repeat stability is not an
acceptance criterion.

## 5. Base training and characterization

Train one Base200 DP checkpoint per task using the same architecture, optimizer,
training seed, epoch/checkpoint rule, and 200-demo input convention.

Evaluate Base200 on frozen train-disjoint environment seeds before collecting
self-improvement rollouts. Report task-level success and capability coverage.
The expected aggregate mean above 50% is a design motivation and monitoring
target, not a reason to tune individual held-out datasets after seeing results.

Base50 is a secondary sensitivity:

- retain/reproduce the official 50-demo DP checkpoint where feasible;
- evaluate it on the same frozen evaluation split;
- report the Base50→Base200 data-scale effect separately;
- do not mix Base50 and Base200 initializations inside the primary causal arms.

## 6. Line A: freeze the method

Line A proceeds in parallel with Base200 collection. Implement and run the
BRACE-v2 intervention pilot only on `place_container_plate`, starting from its
Base200 checkpoint.

Frozen go/no-go rule remains:

- five paired training seeds complete;
- anchor preservation effect point estimate is strictly positive;
- at least four of five paired seeds favor the anchor.

Freeze all method choices before held-out training:

- candidate/branch sampling;
- replay verification thresholds;
- accepted chunk budget;
- random-control matching;
- anchor source and weight;
- optimizer steps and checkpoint selection;
- evaluation splits and stopping rules.

A failed pilot stops confirmatory BRACE claims, but it does not stop Base200
asset collection or archival of diagnostic results.

## 7. Self-generated rollout and verification

For each frozen Base200 policy:

1. collect traced successes and failures on the train-rollout partition;
2. record policy RNG seeds, control traces, dynamic actor state, and branch
   snapshots;
3. replay-audit snapshot restoration and control-trace fidelity;
4. form candidate chunks from both recovery/failure structure and successful
   behavior;
5. branch candidates and matched controls from identical restored states;
6. accept verified chunks using the method frozen in Line A.

Replay/audit failure is task-local. Continue other tasks and record the failed
task. Do not spend open-ended time repairing an individual task unless it is a
systematic harness defect affecting the validity of multiple tasks.

## 8. Confirmatory arms

For each primary held-out task:

| Arm | Continued-training data | Anchor |
| --- | --- | --- |
| Base | none | none |
| U0 | expert-only control | none |
| N1 | matched random chunks | none |
| B1 | branch-verified chunks | none |
| B2 | matched random chunks | frozen anchor |
| B3 | branch-verified chunks | frozen anchor |

Run U0/N1/B1/B2/B3 with training seeds 1–5. Match optimizer steps,
self-generated chunk counts, expert contribution, and nonexpert sampling budget
across causal contrasts. Never let accepted-chunk count silently change training
compute; use the frozen matching rule.

## 9. Evaluation and inference

Use frozen, mutually disjoint splits:

- preservation: repeated evaluation on Base-solved/census behavior;
- confirm-easy: standard `demo_clean` generalization;
- confirm-hard: `demo_randomized` or frozen hard distribution;
- optional Base50 sensitivity on the same evaluation seeds.

Primary contrasts remain:

- anchor main preservation: `mean(B2,B3) - mean(N1,B1)`;
- verification main adaptation: `mean(B1,B3) - mean(N1,B2)`;
- BRACE preservation/adaptation: `B3 - B1`;
- interaction: `(B3-B2) - (B1-N1)`.

Aggregate with task-level weighting and paired hierarchical bootstrap. Report
every preregistered task, including operational failures. Do not pool episodes
as independent task replicates.

## 10. Immediate execution order

1. Let the currently running handover/parity diagnostic finish; archive its JSON
   regardless of outcome and remove it from the launch-gate chain.
2. Stop further 50-demo seed-stability work. Preserve completed 50-demo assets as
   pilot/sensitivity artifacts.
3. Draft and freeze machine-readable `multitask_protocol.v2.json`, task/reserve
   manifest, seed namespaces, and Base200 acquisition rule.
4. Implement/resume native success-first Base200 collection with per-task
   independent status and a 5,000-seed acquisition cap.
5. Train/evaluate Base200 as each task becomes ready; do not wait for all tasks
   before using free training GPUs.
6. In parallel, implement Line A runner and freeze the BRACE-v2 method on
   `place_container_plate` Base200.
7. Start held-out traced rollout/audit for each task only after both its Base200
   asset and the Line A method freeze are ready.
8. Launch the confirmatory arm matrix from an explicit resumable job manifest.
9. Run frozen evaluation, aggregate all tasks, and report Base50 sensitivity
   separately.

## 11. Anti-rabbit-hole checklist

Before starting a new diagnostic or repair, answer:

1. Which primary paper claim or required artifact does it unblock?
2. Does it affect multiple tasks or only make one setup artifact cleaner?
3. Can unrelated tasks continue while this fails?
4. Is there already enough evidence to record the limitation and move on?
5. Would the proposed change use held-out policy outcomes or alter a frozen
   denominator?

If a task-local diagnostic does not affect causal validity, safety, or a required
artifact, archive it and continue the pipeline.
