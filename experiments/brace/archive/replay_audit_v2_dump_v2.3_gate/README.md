# dump A1 replay audit gate (protocol v2.3)

Frozen summary from dump-only `audit-v2` on `experiments/brace/rollouts_traced`.

## Verdict

| Field | Value |
|-------|-------|
| `passed` / `complete` | true |
| `replay_gate_passed` (dump) | **true** |
| restore | 60/60 (100%) |
| replay | 57/60 (95%) |
| `protocol_revision` | 2.3 |
| `git_commit` | 4753527e5b99709e2002f789b4b510f03030d342 |

Failures are deskbin/garbage **translation** only; garbage rotation is excluded from gate under v2.3 symmetry-aware metrics.

## Reproduce

```bash
BRACE_PROTOCOL_V2_PATH=experiments/brace/protocol.v2.3.json \
BRACE_TASKS=dump_bin_bigbin \
BRACE_TRACED_ROLLOUT_DIR=experiments/brace/rollouts_traced \
bash experiments/brace/run_all.sh audit-v2
```

Next: `bash experiments/brace/run_dump_pilot.sh` (A2 branch pilot).
