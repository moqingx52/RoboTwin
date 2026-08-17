# Capability Transport for Pareto-Safe Robot Self-Training

New research line (branch `exp/capability-transport`), started 2026-08-16.
Supersedes the BRACE-RW (Q)-variance route after its E0 v2 gate FAILED on
`place_container_plate` (pooled mean Δ̂_{c=3} = 0.0156 < 0.02 threshold;
exact randomization p = 0.000 passed — variance is real, but it does not
predict a practically meaningful training gain).

Core question (revised 2026-08-17; positioning frozen in
`protocol.t1d.v1.1.json`):

> Given a pretrained robot policy and heterogeneous self-generated
> experience, can small controlled training interventions predict the
> cross-region effects of candidate data sources BEFORE full post-training,
> certify whether a Pareto-safe update exists (hard-region improvement,
> easy/medium non-inferiority), and — when the available data cannot support
> one — correctly abstain and name the minimum targeted expert data Q* that
> restores feasibility?

This is a data attribution / allocation paper, not a success-rate
optimization paper. The estimand is the capability-transport response
T_{gh} = ∂J_g/∂ρ_h over **baseline competence regions** g ∈ {E, M, H} —
π₀ difficulty strata of the frozen T1b census, deliberately NOT claimed to
be semantic skills. Primary results, in order: (1) predictiveness of T̂ on
held-out mixtures, (2) validity of the robust-mixture decision ρ*,
(3) correct rejection on support-degenerate tasks (dump_bin), (4)
interaction/compute efficiency ledger. Final success rate only validates
the decision; it is never the headline.

Relation to adjacent lines: RL fine-tuning (DPPO / VLA-RL) finds a
higher-reward θ assuming cheap online interaction, and cannot say before
training whether an update breaks easy-region competence; we answer the
data question upstream of the optimizer and hand ρ* to SFT (or, post-T3,
to RL). DataMIL attributes per-sample influence on a scalar metric; we
estimate per-source cross-region effects plus a feasibility decision with
a dual certificate. SIME / ReGuide select or generate valuable
trajectories; we decide how much of each source a safe update needs, or
prove no such mixture exists. No RL baseline is on the T1 critical path
(see `protocol.t1d.v1.1.json`, rl_baseline_policy).

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
- T1c v1.2 (`protocol.t1.v1.2.json`, frozen BEFORE any real acquisition):
  v1.1's D_H target of 90 conflated unique-seed feasibility with acquisition
  feasibility — all 15 supported-hard seeds are 1/8 (empirical p̂ = 0.125),
  so P(M_H ≥ 90) ≈ 4.5e-11; the group would have been declared infeasible as
  a design artifact. v1.2 switches D_H to a **fixed opportunity budget**
  (run the frozen 15 × 24-attempt set dry; per-seed success cap 9,
  UnStableError×2 retirement unchanged) judged by the four-part joint gate
  G_H = [U_H ≥ 10] ∧ [n_eff ≥ 10] ∧ [M_H ≥ 36] ∧ [all frozen dose points
  satisfy w = ρ/q ≤ 3], with q_h = M_h/(M_E+M_M+M_H) frozen over the three
  T1c pools only (Base200 rehearsal and D_Q excluded). M_H_min = 36 derives
  from compressing the T1d dose region to ρ_H,max = 0.5 before training.
  G_H = 0 emits the source-infeasibility certificate and halts T1d.
  Monte Carlo record `t1c_dh_mc_feasibility.place_container_plate.v1.json`:
  joint pass 90.7% under empirical p̂, 78.2% under the posterior predictive
  (the ESS part binds when sampled p_s concentrate successes). The old
  posterior-mean dry run is retained as a logic smoke test only.
- T1d amended (`protocol.t1d.v1.1.json`): matched gradient steps S* across
  all 13 dose points (dose = sampling proportion, never training amount);
  ρ_nat (0.9033/0.0855/0.0112, the natural success-only mixture from the
  frozen census) added as 4th held-out point and doubles as the
  Success-only SFT baseline; center point doubles as Difficulty-balanced
  SFT; four primary-result gates preregistered, incl. the dump_bin
  correct-rejection certificate schema and the efficiency ledger.

Legacy BRACE evidence stays read-only under `../brace/` (archives, seeds,
protocols, records). BRACE code was deleted from the working tree in this
branch; it remains available in git history (last full state at commit
8286411 on `exp/dp-self-improvement`).

See `DATA_REUSE.md` for exactly which previously collected data is reusable.
