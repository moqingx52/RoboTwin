# Branch pilot valid run — `dump_bin_bigbin` (protocol v2.3)

Stage 2 matched-continuation pilot for **dump_bin_bigbin** under protocol revision **2.3**.

Cloud run: **2026-08-01**. Rollout source: `experiments/brace/rollouts_traced_pilot/dump_bin_bigbin`.

## Verdict

| Field | Value |
|-------|-------|
| `harness_valid` | **true** |
| `passed` | **true** |
| `matched_success_lift` | **+36.1 pp** |
| `recovery_seed_fraction` | **100%** (4/4 analyzed seeds) |
| `accepted_points` | **7 / 12** |
| `protocol_revision` | 2.3 |
| `git_commit` | `37e2ddf86c93a5e830556ed487ac91f764782738` |

Stage 2 pilot gate: **GO** for `dump_bin_bigbin` under v2.3 harness semantics.

**Caveat**: only **4 / 10** pilot seeds produced branch jobs (failure-chunk filter); pilot-level evidence.

## Analyzed seeds

`100168`, `100075`, `100082`, `100194` — see `analyzed_seeds.json`.

## Authoritative artifacts

Copy from cloud if not present:

- `summary.json`
- `checks.jsonl`
- `protocol.v2.3.json`
- `MANIFEST.sha256` (after `sha256sum` the three files above)

## Written conclusion

[`docs/brace_stage2_dump_pilot_conclusion_20260801.md`](../../../../docs/brace_stage2_dump_pilot_conclusion_20260801.md)

## Cross-task reference

Place pilot GO: `archive/branches_place_pilot_valid_v2.3/`
