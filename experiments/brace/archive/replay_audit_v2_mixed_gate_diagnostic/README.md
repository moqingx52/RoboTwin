# Replay audit v2 mixed-gate diagnostic (superseded)

This directory documents the first complete audit-v2 run that used the **v2.0 mixed
pass-rate gate** (`passed_checks / total_checks` over restore + replay together).

That gate reported `226/240 = 94.17%`, which masked the true replay outcome:

- `restore_determinism`: 120/120 = 100%
- `control_trace_replay`: 106/120 = 88.33%

Per-task replay:

- `place_container_plate`: 59/60 = 98.33%
- `dump_bin_bigbin`: 47/60 = 78.33%

All 14 replay failures were `object_rotation_error` only; robot metrics were tiny.

## Authoritative artifacts on the cloud run

If preserved from the server run, copy these files here:

- `summary.json`
- `checks.jsonl`
- `failures.jsonl`
- `diagnostics.jsonl`

Do **not** treat that run as a formal gate pass/fail under protocol revision 2.1.

## Successor

Use `experiments/brace/protocol.v2.1.json` with separate restore and replay gates.
