#!/usr/bin/env python3
"""Resolve BRACE artifact paths: runs pointers -> archive -> legacy (with warnings)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

AUDIT_ARCHIVE_NAMES = {
    "place_container_plate": "replay_audit_v2_place_v2.3_gate",
    "dump_bin_bigbin": "replay_audit_v2_dump_v2.3_gate",
}

BRANCH_ARCHIVE_NAMES = {
    "branches": "branches_place_pilot_valid_v2.3",
    "branches_confirm": "branches_place_confirm_v2.3",
    "branches_dump": "branches_dump_pilot_valid_v2.3",
}


def _audit_task_gate_passed(summary_path: Path, task: str) -> bool:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return bool(payload.get("tasks", {}).get(task, {}).get("replay_gate_passed", False))


def _audit_archive_summary(task: str, *, brace_dir: Path) -> Path | None:
    archive_name = AUDIT_ARCHIVE_NAMES.get(task)
    if archive_name is None:
        return None
    summary = brace_dir / "archive" / archive_name / "summary.json"
    if summary.is_file() and _audit_task_gate_passed(summary, task):
        return summary
    return None


def resolve_audit_summary(task: str, *, brace_dir: Path = BRACE_DIR) -> Path | None:
    explicit = os.environ.get("BRACE_AUDIT_RUN_DIR")
    if explicit:
        summary = Path(explicit) / "summary.json"
        if not summary.is_file():
            raise ValueError(f"BRACE_AUDIT_RUN_DIR has no summary.json: {explicit}")
        if not _audit_task_gate_passed(summary, task):
            raise ValueError(f"BRACE_AUDIT_RUN_DIR replay gate not passed for {task}: {summary}")
        return summary

    runs = brace_dir / "runs"
    pointer = runs / f"LATEST_AUDIT_{task}"
    if pointer.is_file():
        audit_dir = Path(pointer.read_text(encoding="utf-8").strip())
        summary = audit_dir / "summary.json"
        if summary.is_file():
            if _audit_task_gate_passed(summary, task):
                return summary
            warnings.warn(
                f"Latest audit failed replay gate, skipping immutable run: {audit_dir}",
                stacklevel=2,
            )

    archive = _audit_archive_summary(task, brace_dir=brace_dir)
    if archive is not None:
        return archive

    for legacy in (
        brace_dir / "replay_audit_v2" / task / "summary.json",
        brace_dir / "replay_audit_v2" / "summary.json",
    ):
        if legacy.is_file() and _audit_task_gate_passed(legacy, task):
            warnings.warn(f"Using legacy audit summary (deprecated): {legacy}", stacklevel=2)
            return legacy
    return None


def resolve_branch_dir(label: str, *, brace_dir: Path = BRACE_DIR) -> Path | None:
    runs = brace_dir / "runs"
    pointer = runs / f"LATEST_{label}"
    if pointer.is_file():
        candidate = Path(pointer.read_text(encoding="utf-8").strip())
        if (candidate / "summary.json").is_file():
            return candidate
    archive_name = BRANCH_ARCHIVE_NAMES.get(label)
    archive = brace_dir / "archive" / archive_name if archive_name else None
    if archive is not None and archive.is_dir() and (archive / "summary.json").is_file():
        return archive
    legacy = brace_dir / label
    if (legacy / "summary.json").is_file():
        warnings.warn(f"Using legacy branch dir (deprecated): {legacy}", stacklevel=2)
        return legacy
    return None


def resolve_anchor_smoke_summary(*, brace_dir: Path = BRACE_DIR) -> Path | None:
    pointer = brace_dir / "runs" / "LATEST_anchor_smoke"
    if pointer.is_file():
        run_dir = Path(pointer.read_text(encoding="utf-8").strip())
        summary = run_dir / "summary.json"
        if summary.is_file():
            return summary
    archive = brace_dir / "anchor_smoke" / "summary.json"
    if archive.is_file():
        return archive
    return None


def resolve_dataset_manifest(run_label: str, dataset: str, *, brace_dir: Path = BRACE_DIR) -> Path | None:
    path = brace_dir / "datasets" / f"{run_label}_{dataset}.jsonl"
    if path.is_file():
        return path
    pointer = brace_dir / "runs" / f"LATEST_export_{run_label}"
    if pointer.is_file():
        run_dir = Path(pointer.read_text(encoding="utf-8").strip())
        candidate = run_dir / f"{run_label}_{dataset}.jsonl"
        if candidate.is_file():
            return candidate
    return None


def resolve_seeds_file(name: str, *, brace_dir: Path = BRACE_DIR) -> Path | None:
    pointer = brace_dir / "runs" / f"LATEST_seeds_{name}"
    if pointer.is_file():
        run_dir = Path(pointer.read_text(encoding="utf-8").strip())
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    path = brace_dir / "seeds" / name
    if path.is_file():
        return path
    return None


def resolve_stage_record(stage: str, *, task: str | None = None, brace_dir: Path = BRACE_DIR) -> Path | None:
    from experiments.brace.stage_records import resolve_latest_record

    _ = brace_dir
    return resolve_latest_record(stage, task=task)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "kind",
        choices=["audit-summary", "branch-dir", "anchor-smoke", "dataset", "seeds", "stage-record"],
    )
    parser.add_argument("--task", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--dataset", default="B1")
    parser.add_argument("--seeds-file", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result: Path | None = None
    if args.kind == "audit-summary":
        if not args.task:
            raise SystemExit("--task required")
        result = resolve_audit_summary(args.task)
    elif args.kind == "branch-dir":
        label = args.label or "branches"
        result = resolve_branch_dir(label)
    elif args.kind == "anchor-smoke":
        result = resolve_anchor_smoke_summary()
    elif args.kind == "dataset":
        if not args.run_label:
            raise SystemExit("--run-label required")
        result = resolve_dataset_manifest(args.run_label, args.dataset)
    elif args.kind == "seeds":
        if not args.seeds_file:
            raise SystemExit("--seeds-file required")
        result = resolve_seeds_file(args.seeds_file)
    elif args.kind == "stage-record":
        if not args.label:
            raise SystemExit("--label required (stage name, e.g. audit_v2)")
        result = resolve_stage_record(args.label, task=args.task)

    if result is None:
        print(json.dumps({"found": False}), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"found": True, "path": str(result)}))
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
