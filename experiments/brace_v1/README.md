# BRACE V1 code archive

Frozen at commit `e0e7936b71bd478792053037794c5044090b735c`.

This tree is a **code** archive only. Do not launch production jobs from here.
The live working tree remains [`experiments/brace/`](../brace/).

## Layout

- `ops/` — one-off `_*.py` / `_*.sh` scaffolding moved out of the live tree.
- `snapshot/` — copy of tracked live-tree scripts and protocol JSON at freeze
  (no `_` ops; those live only under `ops/`).

## Data paths (unchanged)

Do not move or rewrite these. V1 and live code still read:

- `experiments/brace/runs/`
- `experiments/brace/rollouts_traced*`
- `experiments/brace/logs/`
- `experiments/brace/archive/` (frozen evidence, not this code archive)
- `experiments/brace/seeds/`
- `experiments/brace/datasets/`

## Ops note

At freeze, PID `1520071` was still running
`bash experiments/brace/_watch_place_base200_pilot.sh`. That process was not
killed. Its argv path is stale after the move; the script itself is now
`experiments/brace_v1/ops/_watch_place_base200_pilot.sh`.

The test helper `behavioral_preservation_go_no_go` now imports from
`experiments.brace_v1.ops._aggregate_place_base200_line_a`.
