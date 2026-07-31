#!/usr/bin/env python3
"""Summarize per-actor replay audit failures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def classify_failure(row: dict) -> str:
    actor_errors = row.get("errors", {}).get("actor_errors", {})
    actor_metric_passed = row.get("actor_metric_passed", {})
    if actor_metric_passed:
        for actor_name, passed in actor_metric_passed.items():
            if passed.get("rotation") is False:
                if actor_name.startswith("garbage_"):
                    return "garbage_rotation"
                if actor_name == "deskbin":
                    return "deskbin_rotation"
                return f"{actor_name}_rotation"
            if passed.get("translation") is False:
                if actor_name.startswith("garbage_"):
                    return "garbage_translation"
                if actor_name == "deskbin":
                    return "deskbin_translation"
                return f"{actor_name}_translation"
    worst_actor = row.get("worst_actor")
    worst_metric = row.get("worst_metric")
    if worst_actor and worst_metric:
        if worst_actor.startswith("garbage_"):
            return f"garbage_{worst_metric}"
        return f"{worst_actor}_{worst_metric}"
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    failures = [
        row
        for row in read_jsonl(args.failures)
        if row.get("check_type") == "control_trace_replay" and not row.get("passed", True)
    ]
    counts = Counter(classify_failure(row) for row in failures)
    garbage_rotation = sum(
        value for key, value in counts.items() if key == "garbage_rotation"
    )
    deskbin_rotation = counts.get("deskbin_rotation", 0)
    garbage_translation = counts.get("garbage_translation", 0)
    deskbin_translation = counts.get("deskbin_translation", 0)
    total = len(failures)

    summary = {
        "total_replay_failures": total,
        "classification_counts": dict(sorted(counts.items())),
        "garbage_rotation_dominates": garbage_rotation >= max(1, total - 1),
        "deskbin_or_garbage_translation": deskbin_translation + garbage_translation,
        "recommendation": (
            "freeze_protocol_v2.2"
            if garbage_rotation >= max(1, total - 1)
            else "continue_deskbin_diagnosis"
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
