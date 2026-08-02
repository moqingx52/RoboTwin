#!/usr/bin/env python3
"""Report branch confirm gate gaps and collection targets for screen.v1.2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

from experiments.brace.replay_audit import read_json, read_jsonl, repo_path


def plan_confirm_collection(*, branch_dir: Path, protocol: dict) -> dict:
    checks = read_jsonl(branch_dir / "checks.jsonl")
    gate = protocol.get("branch_confirm_gate", {})
    min_seeds = int(gate.get("min_seeds", 10))
    min_points = int(gate.get("min_points", 30))
    max_share = float(gate.get("max_seed_accepted_share", 0.2))
    seeds = sorted({int(row["env_seed"]) for row in checks})
    points = len(checks)
    accepted = [row for row in checks if row.get("accepted")]
    per_seed = {}
    for row in accepted:
        per_seed.setdefault(int(row["env_seed"]), 0)
        per_seed[int(row["env_seed"])] += 1
    max_seed_share = max((count / max(1, len(accepted)) for count in per_seed.values()), default=0.0)
    merged_gate = {
        "checks": {
            "min_seeds": len(seeds) >= min_seeds,
            "min_points": points >= min_points,
            "max_seed_accepted_share_le_20pct": max_seed_share <= max_share,
        }
    }
    merged_gate["passed"] = all(merged_gate["checks"].values())
    min_seeds = int(gate.get("min_seeds", 10))
    min_points = int(gate.get("min_points", 30))
    max_share = float(gate.get("max_seed_accepted_share", 0.2))
    seeds = sorted({int(row["env_seed"]) for row in checks})
    points = len(checks)
    accepted = [row for row in checks if row.get("accepted")]
    per_seed = {}
    for row in accepted:
        per_seed.setdefault(int(row["env_seed"]), 0)
        per_seed[int(row["env_seed"])] += 1
    max_seed_share = max((count / max(1, len(accepted)) for count in per_seed.values()), default=0.0)
    return {
        "current": {
            "unique_seeds": len(seeds),
            "total_points": points,
            "accepted_points": len(accepted),
            "max_seed_accepted_share": max_seed_share,
            "merged_gate": merged_gate,
        },
        "targets": {
            "min_seeds": min_seeds,
            "min_points": min_points,
            "max_seed_accepted_share": max_share,
        },
        "gaps": {
            "seeds_needed": max(0, min_seeds - len(seeds)),
            "points_needed": max(0, min_points - points),
            "share_ok": max_seed_share <= max_share,
        },
        "recommendation": (
            "Collect additional independent env seeds before exporting the next B1 manifest."
            if len(seeds) < min_seeds or points < min_points or max_seed_share > max_share
            else "Confirm gate thresholds are satisfied."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-dir", type=Path, default=Path("experiments/brace/branches"))
    parser.add_argument("--protocol", type=Path, default=Path("experiments/brace/screen_protocol.v1.2.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = read_json(repo_path(args.protocol))
    payload = plan_confirm_collection(branch_dir=repo_path(args.branch_dir), protocol=protocol)
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
