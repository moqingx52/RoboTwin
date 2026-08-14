#!/usr/bin/env python3
import json
from pathlib import Path

root = Path("experiments/brace/runs/base200_frozen_eval_20260811T012027Z/results")
if root.is_dir():
    for path in sorted(root.glob("*/*.json")):
        if "_shard_" in path.name:
            continue
        payload = json.loads(path.read_text())
        progress = payload.get("progress") or {}
        heldout = (payload.get("splits") or {}).get("id_heldout") or {}
        print(
            path.parent.name,
            path.name,
            "complete",
            bool(progress.get("complete")),
            "rows",
            len(payload.get("rows") or []),
            "sr",
            heldout.get("mean_sr"),
        )

print("seed_manifests", sorted(p.name for p in Path("experiments/brace/seeds/multitask_v1").glob("*.json")))
