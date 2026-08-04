# v1.4.1 Confirmatory Census Feasibility Failure (2026-08-03)

## Summary

P1a base census completed successfully under `screen.v1.4.1.confirmatory_preservation.json`, but P1b cohort selection failed the frozen `min_untouched_base_solved=60` gate.

| Stage | Run | Outcome |
|-------|-----|---------|
| P1a | `experiments/brace/runs/20260803T144820Z_confirmatory_base_census_place_container_plate` | complete, 600 episodes, offset 3000 |
| P1b | `experiments/brace/runs/20260803T235334Z_select_preservation_cohort_place_container_plate` | **failed**, `untouched_n=42` |

## Root cause

After applying frozen exclusions (anchor, SFT chunk, branch pilot/confirm, Phase 3C behavior seeds), only **42** seeds met the integer **2-of-3** enrollment rule on the 100-seed census `id_heldout` pool. This is below the protocol minimum of 60.

With n=42, the zero-event 95% upper bound is approximately **6.88%**, which exceeds the frozen preservation gate target of **<5%**. At least **59** untouched seeds are required for that bound; retaining n=60 is appropriate.

## Resolution (v1.4.2)

Amendment `screen.v1.4.2.confirmatory_preservation.json` expands the census candidate pool to **200** seeds (`eval_id` + unused `reserve`) while keeping:

- adaptation `id_heldout` at the original **100** eval seeds
- offset 3000 / 4000 isolation
- 2-of-3 integer enrollment
- `min_untouched=60`
- all exclusion sources and C0/C1 training seeds

Rationale: enrollment feasibility expansion based on base-only census evidence, **before** observing any C0/C1 training results.

## Artifacts preserved

- Failed P1b cohort JSON: `place_container_plate_preservation_cohort.json` (frozen=true, meets_min_untouched=false)
- `LATEST_preservation_cohort` was **not** written (fail-closed by design)
- P1c/P1d were **not** started
