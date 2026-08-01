# Branch confirmatory run — `place_container_plate` (protocol v2.3)

Held-out confirmatory Stage 2 branch for **place_container_plate** (Track B).

Cloud run: **2026-08-02**. Rollout source: `experiments/brace/rollouts_traced_pilot/place_container_plate` (confirm 5 seeds).

## Verdict

| Field | Value |
|-------|-------|
| Single-batch `passed` | **true** |
| `harness_valid` | **true** |
| `matched_success_lift` | **+29.6 pp** |
| `recovery_seed_fraction` | **100%** (1/1 analyzed seed) |
| `accepted_points` | **1 / 3** |
| **Merged confirmatory gate** | **NO-GO** (6 seeds, 18 points) |

## Confirm seeds (5 selected, 1 analyzed)

Selected: `100100, 100043, 100081, 100020, 100071`  
Analyzed in branch: **`100071` only**

## Artifacts

- `summary.json`, `checks.jsonl` — copy from `experiments/brace/branches_confirm/`
- `merged_gate.json` — merge evaluation vs pilot archive

## Written conclusion

[`docs/brace_track_bc_confirm_export_20260802.md`](../../../../docs/brace_track_bc_confirm_export_20260802.md)
