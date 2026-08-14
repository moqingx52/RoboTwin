# BRACE V2 anchor-route code archive

Frozen from branch `exp/dp-self-improvement` at commit
`94f7f94` (all scripts byte-identical to `experiments/brace_v1/snapshot/`,
which was frozen at `e0e7936`; no functional edits happened between the two).

This tree is a **code** archive only. Do not launch production jobs from here.
The live working tree remains [`experiments/brace/`](../brace/).

## Why archived

The single-timestep denoising-MSE anchor route (anchor smoke → feasibility →
calibration → behavior eval) is superseded by the **BRACE-RW** plan
(return-weighted denoising loss + E0 variance gate + trajectory-KL `D_path`
capability preservation). Rationale:

- `brace_v2_derivation.md` §3: dual ascent needs λ*≈55 but reaches ≈0.025 in
  the 2,470-step budget (three orders of magnitude short); the 16-window
  anchor set is interpolation, not a distribution constraint; anchor gradients
  are near-collinear with expert rehearsal (P5), so marginal value ≈ 0.
- The BRACE-RW review (2026-08): exact local improvement
  Δ_c(s) = c·Var_a[Q₀]/(1+cV₀) from weight w=(1+cG)/(1+cV₀); capability
  preservation moves to full-trajectory KL, making the single-timestep anchor
  obsolete. Plan: `experiments/brace/BRACE_RW_PLAN.md`.

## Archived scripts (14)

Anchor route: `anchor_smoke.py`, `anchor_unit_smoke.py`,
`anchor_training_smoke.py`, `anchor_feasibility_test.py`,
`anchor_feasibility_diagnostic.py`, `anchor_feasibility_runner.py`,
`anchor_calibration_runner.py`, `anchor_diagnostic_loop.py`,
`anchor_grad_diagnostics.py`, `anchor_probe_eval.py`, `anchor_probe_split.py`,
`build_anchor_replay_set.py`, `orchestrate_behavior_eval.py`,
`test_anchor_grad_diagnostics.py`.

Intra-archive imports were rewritten to `experiments.brace_v2_legacy.*`.
Imports of still-live modules (`orchestrate`, `replay_audit*`,
`anchor_behavior_eval`, `build_screen_dataset`, `preservation_groups`,
`screen_gates`, `stage_records`) intentionally still point at
`experiments.brace.*`.

## Kept live (do not archive)

- `experiments/brace/anchor_behavior_eval.py` — imported by
  `confirmatory_base_census.py`, `confirmatory_preservation_eval.py`,
  `select_preservation_cohort.py`.
- `experiments/brace/orchestrate_calibration.py` — invoked by
  `run_all.sh confirmatory-preservation`; its worker path now points at
  `brace_v2_legacy/anchor_calibration_runner.py`.
- `experiments/brace/preservation_sampler.py` — imported by
  `policy/DP/.../robotworkspace.py`.

## Live pointers into this archive

- `experiments/brace/orchestrate_calibration.py:52` → `anchor_calibration_runner.py`
- `experiments/brace/orchestrate.py` → gate helpers from `anchor_smoke.py`,
  prepare:anchor job → `build_anchor_replay_set.py`
- `experiments/brace/finetune.sh` → `anchor_feasibility_test.py`
- `experiments/brace/test_track_c_scaffold.py` → several archived modules

`run_all.sh` stages `anchor-smoke`, `anchor-feasibility`, `anchor-calibration`,
`anchor-behavior-eval` now exit with a redirect notice.

## Data paths (unchanged)

Do not move or rewrite these. Archived and live code still read:

- `experiments/brace/runs/` (incl. `LATEST_anchor_*` pointers)
- `experiments/brace/rollouts_traced*`
- `experiments/brace/logs/`
- `experiments/brace/archive/` (frozen evidence, not this code archive)
- `experiments/brace/seeds/`
- `experiments/brace/datasets/`
