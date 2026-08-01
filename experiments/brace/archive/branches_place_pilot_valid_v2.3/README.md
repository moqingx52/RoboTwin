# Branch pilot valid run (protocol v2.3)

This directory archives the **valid** `place_container_plate` Stage-2 branch pilot
collected under protocol revision **2.3** after chunk-boundary harness alignment fixes.

Cloud run: 2026-08-01. Rollout source:
`experiments/brace/rollouts_traced_pilot/place_container_plate`.

## Verdict

| Field | Value |
|-------|-------|
| `harness_valid` | **true** |
| `passed` | **true** |
| `complete` | true |
| `protocol_revision` | 2.3 |
| `git_commit` | 4753527e5b99709e2002f789b4b510f03030d342 |

## Task summary (`place_container_plate`)

| Metric | Value |
|--------|-------|
| Branch points evaluated | 15 (5 seeds × 3 snapshots) |
| Accepted points (LCB > 0.15) | 11 / 15 |
| Matched success lift | **+62.2 pp** |
| Recovery seed fraction | **100%** (5/5 seeds) |
| Pilot gate threshold | lift ≥ +20 pp **or** recovery ≥ 20% |

Stage 2 pilot gate: **GO** for `place_container_plate` under v2.3 harness semantics.

## Comparison to invalid v2.2 pilot

| Metric | v2.2 invalid | v2.3 valid |
|--------|--------------|------------|
| Archive | `archive/branches_place_pilot_invalid_v2.2/` | this directory |
| `harness_valid` | false | true |
| Candidate baseline | 0/120 | non-zero (harness sanity passed) |
| `matched_success_lift` | -16.4% | +62.2% |
| `accepted_points` | 0/40 | 11/15 |

The v2.2 run is **not** interpretable as model failure; see invalid archive README.

## Harness semantics (v2.3)

- Branch points: real `branch_snapshots` only (`snapshot_id` deduped).
- Restore → replay success control trace to **next chunk boundary** → inject full chunk.
- `policy_chunk_index` from `control_steps[]` (no `len(actions)` heuristic).
- Controls: same `chunk_index`, K=3 distinct failure rollouts per seed.
- Runtime counters restored: `physics_step`, `take_action_cnt`, `policy_chunk_index`.

## Authoritative artifacts

| File | Role |
|------|------|
| `summary.json` | Frozen branch gate summary |
| `checks.jsonl` | Per-rollout branch rows (copy from cloud if absent) |
| `protocol.v2.3.json` | Frozen branch protocol |
| `analyzed_seeds.json` | Five seeds used in the 15 evaluated points (Track B exclusion) |
| `MANIFEST.sha256` | SHA256 of `checks.jsonl`, `summary.json`, `protocol.v2.3.json` |

Regenerate manifest on cloud after copying `checks.jsonl`:

```bash
bash experiments/brace/archive_place_pilot.sh
```

## MANIFEST.sha256

```
acac9d34154a626a18f892779e20966d8ae7e68e33253bb28d3aaea6cc2d85c8  summary.json
d1692a09b49fe9137918bb644f096d75593e15f1e63298f4f4d4bf0dc35e5d1d  protocol.v2.3.json
```

After copying `checks.jsonl` from cloud, rerun `bash experiments/brace/archive_place_pilot.sh` to include it in the manifest.

## Reproduce

```bash
BRACE_PROTOCOL_V2_PATH=experiments/brace/protocol.v2.3.json \
BRACE_TASKS=place_container_plate \
BRACE_TRACED_ROLLOUT_DIR=experiments/brace/rollouts_traced_pilot \
bash experiments/brace/run_all.sh audit-v2

BRACE_PROTOCOL_V2_PATH=experiments/brace/protocol.v2.3.json \
BRACE_TASKS=place_container_plate \
BRACE_TRACED_ROLLOUT_DIR=experiments/brace/rollouts_traced_pilot \
bash experiments/brace/run_all.sh branch
```

## Written conclusion

See [docs/brace_stage2_place_pilot_conclusion.md](../../../docs/brace_stage2_place_pilot_conclusion.md).

## Superseded invalid run

`experiments/brace/archive/branches_place_pilot_invalid_v2.2/`
