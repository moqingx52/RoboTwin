# Capability Transport for Pareto-Safe Robot Self-Training

New research line (branch `exp/capability-transport`), started 2026-08-16.
Supersedes the BRACE-RW (Q)-variance route after its E0 v2 gate FAILED on
`place_container_plate` (pooled mean Δ̂_{c=3} = 0.0156 < 0.02 threshold;
exact randomization p = 0.000 passed — variance is real, but it does not
predict a practically meaningful training gain).

Core question:

> Can a small-dose "tomography" of training-data sources estimate a
> capability-transport matrix T_{gh} (effect of data source h on difficulty
> group g), certify before full fine-tuning whether a Pareto-safe update
> exists (hard-group improvement, easy/medium non-inferiority), and — when
> infeasible — derive the minimum targeted expert data Q* that restores
> feasibility?

Three theory targets:

1. Coverage-collapse law of success-only self-training
   (μ_g^(K)/μ_h^(K) = μ_g^(0)/μ_h^(0) · Π_k p_g^(k)/p_h^(k)).
2. Unidentifiability of zero-support hard regions under success-only replay.
3. Local safe-update theorem via robust mixture optimization over a
   confidence set 𝒰_T, with dual certificates naming the missing data type.

Phases:

| Phase | Content | Method changes allowed |
| --- | --- | --- |
| T0 | Audit of legacy BRACE results (motivation only) | — |
| T1 | Tomography development on `place_container_plate`, `dump_bin_bigbin`: 9-point D-optimal small-dose design over D_E/D_M/D_H/D_Q, 4 training seeds/point, 10–20% budget | yes |
| T2 | Freeze difficulty definition, data sources, solver, statistical gates | freeze point |
| T3 | Confirmatory 10 held-out tasks (reuse `../brace/multitask_tasks.v1.json` task list) | no |
| T4 | Architecture replication (ACT or DP3, 3 pre-specified tasks), only if T3 passes | no |

Key invariants carried over from the BRACE audit:

- Per-sample overuse bound w = ρ/q ≤ 3 (avoids the old ~900× chunk reuse).
- Difficulty is frozen from independent baseline rollouts of π₀
  (Beta–Binomial soft groups E [0.7,1], M [0.3,0.7), H [0,0.3), plus a 0/8
  unsupported tail); never regrouped after training.
- Hierarchical inference task → training seed → env seed → policy repeat,
  tasks equally weighted.
- Main gate: Γ = min{Δ_H − 0.05, Δ_E + 0.05, Δ_M + 0.05}, one-sided 95%
  lower confidence bound > 0; plus Ours vs Random-Q hard-group superiority,
  ≥ 8/10 tasks positive on hard, no task degrading E/M by > 10 pp.

Status update 2026-08-17 (see `t1c_source_feasibility.v1.json` and
`protocol.t1.v1.1.json`):

- T1b π₀ difficulty measurement is complete and frozen (v1) for both tasks.
- `dump_bin_bigbin` FAILED the pre-training source-feasibility gate:
  environment validity 112/240 (128 seeds structurally fail setup with
  UnStableError, 0/8 evaluable), medium group = 3 unique env seeds,
  supported-hard = 8. The standard E/M/H tomography is not identifiable
  there; the task is reassigned as the support-degenerate stress case for
  theory target 2. v1 seeds/groups are never swapped or rerun.
- `place_container_plate` PASSED (medium 31, supported-hard 15) and proceeds
  under the v1.1 T1c amendment: group-balanced, frozen-budget round-robin
  success-first acquisition (candidate stream + budgets frozen, realized
  successful seeds are an outcome), replacing v1's proportional-100-seed
  rule. Population metrics still use true group prevalences as weights.

Legacy BRACE evidence stays read-only under `../brace/` (archives, seeds,
protocols, records). BRACE code was deleted from the working tree in this
branch; it remains available in git history (last full state at commit
8286411 on `exp/dp-self-improvement`).

See `DATA_REUSE.md` for exactly which previously collected data is reusable.
