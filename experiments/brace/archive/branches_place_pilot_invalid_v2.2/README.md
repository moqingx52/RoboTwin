# Branch pilot invalid run (protocol v2.2)

This directory archives the **invalid** `place_container_plate` Stage-2 branch pilot
collected under protocol revision **2.2** before harness time-alignment fixes (v2.3).

## Verdict

| Field | Value |
|-------|-------|
| `harness_valid` | **false** |
| `passed` | false (not interpretable as model failure) |
| `candidate_success_rate` | 0/120 (systematic harness misalignment) |
| `protocol_revision` | 2.2 |
| `git_commit` | 38849b895515f420c19d5d15272fb79d1706f798 |

## Known harness bugs (fixed in v2.3)

1. Candidate chunk selected via `len(actions)` heuristic with fallback to last chunk.
2. Branch points claimed arbitrary `physics_step` while restoring nearest quartile snapshot.
3. Snapshots taken mid-chunk; branch injected from chunk start without replay-to-boundary.
4. `restore_branch_snapshot` did not restore `take_action_cnt` / `physics_step` budget.
5. Controls sampled from random success-trace chunk indices, applied to failure traces.

## Diagnostic command

```bash
jq -s '
  map(select(.branch_role=="candidate" and .continuation_seed==0))
  | group_by([.env_seed, .snapshot_id])
  | map({
      seed: .[0].env_seed,
      snapshot_id: .[0].snapshot_id,
      claimed_steps: (map(.physics_step) | unique),
      chunks: (map(.chunk_index) | unique)
    })
  | map(select((.claimed_steps | length) > 1))
' checks.jsonl
```

Non-empty output proves the same snapshot was reused for multiple claimed branch steps.

## Successor

See `experiments/brace/protocol.v2.3.json` and the v2.3 branch harness in
`experiments/brace/collect_branches.py`.
