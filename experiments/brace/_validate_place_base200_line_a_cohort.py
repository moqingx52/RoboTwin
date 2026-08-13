#!/usr/bin/env python3
"""Validate a frozen Base200 Line A preservation cohort before preservation eval.

Checks frozen=true, meets_min_untouched=true, and pairwise seed disjointness
across cohort partitions and material exclusions. Does not start eval.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
BRACE = REPO / "experiments" / "brace"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.brace.replay_audit import read_json, write_json_atomic

COHORT_KEYS = ("untouched_preservation", "anchor_train", "anchor_probe", "boundary")


def resolve_cohort_path(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit if explicit.is_absolute() else REPO / explicit
        if not path.is_file():
            raise SystemExit(f"missing cohort: {path}")
        return path
    pointer = BRACE / "runs/LATEST_place_base200_line_a_preservation_cohort"
    if not pointer.is_file():
        raise SystemExit(f"missing cohort pointer: {pointer}")
    path = Path(pointer.read_text(encoding="utf-8").strip())
    if not path.is_absolute():
        path = REPO / path
    if not path.is_file():
        raise SystemExit(f"cohort pointer target missing: {path}")
    return path


def as_int_set(values: Any) -> set[int]:
    if not values:
        return set()
    return {int(v) for v in values}


def validate_cohort(payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not payload.get("frozen"):
        errors.append("frozen must be true")
    if not payload.get("meets_min_untouched"):
        errors.append("meets_min_untouched must be true")
    cohorts = payload.get("cohorts") or {}
    sets: dict[str, set[int]] = {}
    for key in COHORT_KEYS:
        if key not in cohorts:
            errors.append(f"missing cohort partition: {key}")
            continue
        sets[key] = as_int_set(cohorts[key])
    overlaps: list[dict[str, Any]] = []
    keys = list(sets)
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            shared = sorted(sets[left] & sets[right])
            if shared:
                overlaps.append({"left": left, "right": right, "shared_count": len(shared), "shared_sample": shared[:20]})
                errors.append(f"seed overlap between {left} and {right}: n={len(shared)}")
    exclusions = payload.get("exclusions") or {}
    excluded_union: set[int] = set()
    for name, values in exclusions.items():
        excluded_union |= as_int_set(values)
    untouched = sets.get("untouched_preservation", set())
    leaked = sorted(untouched & excluded_union)
    if leaked:
        errors.append(f"untouched_preservation intersects exclusions: n={len(leaked)}")
    if payload.get("run_type") and payload.get("run_type") != "base200_line_a_preservation_cohort":
        warnings.append(f"unexpected run_type={payload.get('run_type')}")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "partition_sizes": {k: len(v) for k, v in sets.items()},
        "overlap_details": overlaps,
        "untouched_exclusion_leak_sample": leaked[:20],
        "excluded_union_count": len(excluded_union),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=None)
    parser.add_argument(
        "--write-report",
        type=Path,
        default=None,
        help="Optional JSON report path (default: beside cohort).",
    )
    args = parser.parse_args()
    cohort_path = resolve_cohort_path(args.cohort)
    payload = read_json(cohort_path)
    check = validate_cohort(payload)
    report = {
        "schema_version": 1,
        "kind": "place_base200_line_a_cohort_validation",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cohort_path": str(cohort_path),
        "cohort_frozen": bool(payload.get("frozen")),
        "meets_min_untouched": bool(payload.get("meets_min_untouched")),
        "protocol_path": payload.get("protocol_path"),
        "protocol_revision": payload.get("protocol_revision"),
        **check,
    }
    out = args.write_report
    if out is None:
        out = cohort_path.with_name(cohort_path.stem + "_validation.json")
    elif not out.is_absolute():
        out = REPO / out
    write_json_atomic(out, report)
    print(json.dumps(report, indent=2))
    return 0 if check["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
