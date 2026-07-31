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


def resolve_failures_path(path: Path | None, audit_dir: Path | None) -> Path:
    if audit_dir is not None:
        candidate = audit_dir / "failures.jsonl"
        if candidate.is_file():
            return candidate
        summary_path = audit_dir / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            artifact = summary.get("artifacts", {}).get("failures", "failures.jsonl")
            candidate = audit_dir / artifact
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"no failures.jsonl under audit dir: {audit_dir}")

    if path is None:
        raise ValueError("provide --failures <path> or --audit-dir <dir>")

    candidates = [path]
    if path.suffix == ".json":
        candidates.append(path.with_suffix(".jsonl"))
    if not path.suffix:
        candidates.append(path.with_suffix(".jsonl"))
    candidates.append(path.parent / "failures.jsonl")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"could not find failures file for {path}; expected failures.jsonl next to summary.json"
    )


def classify_failure(row: dict) -> str:
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

    metric_passed = row.get("metric_passed", {})
    actor_errors = row.get("errors", {}).get("actor_errors", {})
    if actor_errors:
        for actor_name in sorted(actor_errors):
            metrics = actor_errors[actor_name]
            rotation_failed = (
                metric_passed.get("object_rotation_error") is False
                and float(metrics["rotation_error"]) >= float(row.get("thresholds", {}).get("object_rotation_error", 0.05))
            )
            translation_failed = (
                metric_passed.get("object_translation_error") is False
                and float(metrics["translation_error"]) >= float(row.get("thresholds", {}).get("object_translation_error", 0.005))
            )
            if rotation_failed:
                if actor_name.startswith("garbage_"):
                    return "garbage_rotation"
                if actor_name == "deskbin":
                    return "deskbin_rotation"
                return f"{actor_name}_rotation"
            if translation_failed:
                if actor_name.startswith("garbage_"):
                    return "garbage_translation"
                if actor_name == "deskbin":
                    return "deskbin_translation"
                return f"{actor_name}_translation"
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--failures",
        type=Path,
        default=None,
        help="Path to failures.jsonl (also accepts failures.json and resolves to .jsonl).",
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=None,
        help="Replay audit output directory containing failures.jsonl and summary.json.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    failures_path = resolve_failures_path(args.failures, args.audit_dir)
    failures = [
        row
        for row in read_jsonl(failures_path)
        if row.get("check_type") == "control_trace_replay" and not row.get("passed", True)
    ]
    counts = Counter(classify_failure(row) for row in failures)
    garbage_rotation = counts.get("garbage_rotation", 0)
    deskbin_rotation = counts.get("deskbin_rotation", 0)
    garbage_translation = counts.get("garbage_translation", 0)
    deskbin_translation = counts.get("deskbin_translation", 0)
    total = len(failures)
    rotation_only_garbage = (
        total > 0
        and deskbin_rotation == 0
        and deskbin_translation == 0
        and garbage_translation == 0
        and garbage_rotation == total
    )

    summary = {
        "failures_path": str(failures_path),
        "total_replay_failures": total,
        "classification_counts": dict(sorted(counts.items())),
        "garbage_rotation_failures": garbage_rotation,
        "deskbin_rotation_failures": deskbin_rotation,
        "garbage_translation_failures": garbage_translation,
        "deskbin_translation_failures": deskbin_translation,
        "garbage_rotation_dominates": garbage_rotation >= max(1, total - 1),
        "rotation_only_garbage_failures": rotation_only_garbage,
        "deskbin_or_garbage_translation": deskbin_translation + garbage_translation,
        "recommendation": (
            "freeze_protocol_v2.2"
            if rotation_only_garbage or garbage_rotation >= max(1, total - 1)
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
