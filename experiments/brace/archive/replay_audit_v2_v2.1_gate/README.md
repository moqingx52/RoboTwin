# Replay audit v2 protocol v2.1 gate run

This directory archives the first complete audit-v2 run under **protocol revision 2.1**
with separate restore and replay gates.

## Gate semantics (v2.1)

- `restore_determinism`: 120/120 required (100%)
- `control_trace_replay` (global): >= 95%
- `per_task control_trace_replay`: >= 95% for each task
- Overall `passed` requires all of the above plus `complete=true`

Mixed pass-rate (`passed_checks / total_checks` over restore + replay) is recorded as
`pass_rate` for diagnostics only and does **not** drive the gate.

## Results (cloud run, 2026-07-31)

| Metric | Value |
|--------|-------|
| `complete` | true |
| `passed` | false |
| Mixed pass rate | 226/240 = 94.17% |
| `restore_determinism` | 120/120 = 100% |
| `control_trace_replay` | 106/120 = 88.33% |

Per-task replay:

| Task | Replay pass rate | `replay_gate_passed` |
|------|------------------|----------------------|
| `place_container_plate` | 59/60 = 98.33% | true |
| `dump_bin_bigbin` | 47/60 = 78.33% | false |

All 14 replay failures were `object_rotation_error` only; robot metrics were tiny.

### Failure trajectories (object_rotation_error)

| task | env_seed | rollout_id | snapshot_id | object_rotation_error |
|------|----------|------------|-------------|----------------------|
| place_container_plate | 100023 | 2 | 2 | 0.0503 |
| dump_bin_bigbin | 100038 | 1 | 1 | 0.0509 |
| dump_bin_bigbin | 100038 | 1 | 2 | 0.1131 |
| dump_bin_bigbin | 100038 | 3 | 1 | 0.0616 |
| dump_bin_bigbin | 100038 | 3 | 2 | 0.1080 |
| dump_bin_bigbin | 100041 | 5 | 1 | 0.0581 |
| dump_bin_bigbin | 100041 | 5 | 2 | 0.0849 |
| dump_bin_bigbin | 100042 | 2 | 1 | 0.0518 |
| dump_bin_bigbin | 100042 | 2 | 2 | 0.0732 |
| dump_bin_bigbin | 100042 | 6 | 2 | 0.1186 |
| dump_bin_bigbin | 100136 | 3 | 1 | 0.0738 |
| dump_bin_bigbin | 100136 | 3 | 2 | 0.0793 |
| dump_bin_bigbin | 100235 | 5 | 1 | 0.1120 |
| dump_bin_bigbin | 100235 | 5 | 2 | 0.1103 |

## Authoritative artifacts

If preserved from the server run, copy these files here:

- `summary.json`
- `checks.jsonl`
- `failures.jsonl`
- `diagnostics.jsonl`

## Successor

Per-actor diagnostics and symmetry-aware `protocol.v2.2.json` (sphere garbage actors
skip rotation checks). See `experiments/brace/protocol.v2.2.json`.
