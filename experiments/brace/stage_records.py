#!/usr/bin/env python3
"""Emit timestamped BRACE stage metadata into experiments/brace/records/.

Machine-readable experiment index (gate status, paths, highlights).
Human conclusions stay in docs/; frozen evidence stays in archive/.
"""

from __future__ import annotations

import json
import argparse
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
RECORDS_DIR = BRACE_DIR / "records"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.replay_audit import git_commit, read_json, repo_path, write_json_atomic


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(*parts: str) -> str:
    return "_".join(part.replace("/", "_").replace(" ", "_") for part in parts if part)


def find_run_dir(artifact_dir: Path) -> Path | None:
    for parent in (artifact_dir, artifact_dir.parent, artifact_dir.parent.parent):
        if (parent / "meta.json").is_file():
            return parent
    return None


def load_run_meta(run_dir: Path | None) -> dict[str, Any] | None:
    if run_dir is None:
        return None
    meta_path = run_dir / "meta.json"
    if meta_path.is_file():
        return read_json(meta_path)
    return None


def audit_highlights(summary: dict[str, Any]) -> dict[str, Any]:
    replay = summary.get("control_trace_replay", {})
    return {
        "passed": summary.get("passed"),
        "complete": summary.get("complete"),
        "restore_passed": summary.get("restore_determinism", {}).get("passed"),
        "replay_passed": replay.get("passed"),
        "replay_total_checks": replay.get("total_checks"),
        "replay_pass_rate": replay.get("pass_rate"),
        "tasks": {
            task: {
                "replay_gate_passed": stats.get("replay_gate_passed"),
            }
            for task, stats in summary.get("tasks", {}).items()
        },
    }


def branch_highlights(summary: dict[str, Any]) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for task, stats in summary.get("tasks", {}).items():
        tasks[task] = {
            "passed": stats.get("passed"),
            "accepted_points": stats.get("accepted_points"),
            "total_points": stats.get("total_points"),
            "matched_success_lift": stats.get("matched_success_lift"),
        }
    return {
        "passed": summary.get("passed"),
        "complete": summary.get("complete"),
        "harness_valid": summary.get("harness_valid"),
        "tasks": tasks,
    }


def export_highlights(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_label": summary.get("run_label"),
        "task": summary.get("task"),
        "b1_rows": summary.get("b1_rows"),
        "n1_rows": summary.get("n1_rows"),
    }


def anchor_highlights(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": summary.get("passed"),
        "checks": summary.get("checks"),
    }


def promote_highlights(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "target": payload.get("target"),
        "source_run": payload.get("source_run"),
        "archive_dir": payload.get("archive_dir"),
        "artifact_count": len(payload.get("copied", [])),
    }


def inventory_highlights(inventory: dict[str, Any]) -> dict[str, Any]:
    missing = [
        entry["repo_path"]
        for entry in inventory.get("artifacts", [])
        if entry.get("required") and not entry.get("exists")
    ]
    return {
        "artifact_count": len(inventory.get("artifacts", [])),
        "missing_required": missing,
        "bundle_summary": inventory.get("bundle_summary"),
    }


def extract_highlights(stage: str, summary: dict[str, Any]) -> dict[str, Any]:
    if stage in {"audit_v2", "archive_replay_gate"}:
        return audit_highlights(summary)
    if stage.startswith("branch") or stage == "merged_gate":
        return branch_highlights(summary)
    if stage.startswith("export"):
        return export_highlights(summary)
    if stage == "anchor_smoke":
        return anchor_highlights(summary)
    if stage == "promote_run":
        return promote_highlights(summary)
    if stage == "artifact_inventory":
        return inventory_highlights(summary)
    return {
        "passed": summary.get("passed"),
        "complete": summary.get("complete"),
    }


def record_filename(
    *,
    run_id: str | None,
    stage: str,
    tasks: list[str] | None = None,
    label: str | None = None,
) -> str:
    stamp = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if label and tasks:
        return f"{stamp}_{slugify(stage, label, *tasks)}.json"
    if label:
        return f"{stamp}_{slugify(stage, label)}.json"
    if tasks and len(tasks) == 1:
        return f"{stamp}_{slugify(stage, tasks[0])}.json"
    if tasks:
        return f"{stamp}_{slugify(stage, *tasks)}.json"
    return f"{stamp}_{slugify(stage)}.json"


EVIDENCE_SUFFIXES = {".json", ".jsonl", ".sha256"}
EVIDENCE_NAMES = {"MANIFEST.sha256"}


def _portable_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def snapshot_evidence(artifact_dir: Path, bundle_dir: Path) -> list[dict[str, Any]]:
    """Copy small machine-readable evidence so records/ is self-contained.

    Binary rollouts/checkpoints are intentionally excluded. Their paths remain in
    summaries and run metadata and must be backed up separately when required.
    """
    copied: list[dict[str, Any]] = []
    if not artifact_dir.is_dir():
        return copied
    for source in sorted(artifact_dir.rglob("*")):
        if not source.is_file():
            continue
        if source.name not in EVIDENCE_NAMES and source.suffix not in EVIDENCE_SUFFIXES:
            continue
        relative = source.relative_to(artifact_dir)
        destination = bundle_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(
            {
                "path": str(relative),
                "size_bytes": destination.stat().st_size,
            }
        )
    return copied


def append_index(record_path: Path, record: dict[str, Any]) -> None:
    index_path = RECORDS_DIR / "index.jsonl"
    try:
        rel_path = str(record_path.relative_to(REPO_ROOT))
    except ValueError:
        rel_path = str(record_path)
    line = {
        "record_id": record["record_id"],
        "stage": record["stage"],
        "created_at": record["created_at"],
        "path": rel_path,
        "passed": record.get("gate", {}).get("passed"),
    }
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, ensure_ascii=False) + "\n")


def update_latest_pointer(stage: str, record_path: Path, *, task: str | None = None) -> None:
    pointer = RECORDS_DIR / f"LATEST_{stage}"
    if task:
        pointer = RECORDS_DIR / f"LATEST_{stage}_{task}"
    pointer.write_text(str(record_path) + "\n", encoding="utf-8")


def emit_stage_record(
    stage: str,
    *,
    summary_path: Path | None = None,
    summary: dict[str, Any] | None = None,
    tasks: list[str] | None = None,
    label: str | None = None,
    run_dir: Path | None = None,
    artifact_dir: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write one timestamped metadata record; return the record path."""
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)

    if summary is None and summary_path is not None:
        summary = read_json(repo_path(summary_path))
    if summary is None:
        summary = {}

    if artifact_dir is not None:
        artifact_dir = repo_path(artifact_dir)
    else:
        artifact_dir = repo_path(summary_path).parent if summary_path else (run_dir or RECORDS_DIR)
    if run_dir is None:
        run_dir = find_run_dir(artifact_dir)
    meta = load_run_meta(run_dir)
    run_id = meta.get("run_id") if meta else None
    if tasks is None and meta:
        tasks = list(meta.get("tasks", []))

    filename = record_filename(run_id=run_id, stage=stage, tasks=tasks, label=label)
    record_path = RECORDS_DIR / filename
    bundle_dir = RECORDS_DIR / "bundles" / filename.removesuffix(".json")
    evidence = []
    if artifact_dir.resolve() != RECORDS_DIR.resolve():
        evidence = snapshot_evidence(artifact_dir, bundle_dir)
    highlights = extract_highlights(stage, summary)
    gate_passed = highlights.get("passed")
    if gate_passed is None and "missing_required" in highlights:
        gate_passed = not highlights["missing_required"]

    record: dict[str, Any] = {
        "schema_version": 1,
        "record_id": filename.removesuffix(".json"),
        "stage": stage,
        "tasks": tasks or [],
        "label": label,
        "created_at": utc_now(),
        "hostname": socket.gethostname(),
        "git_commit": git_commit(),
        "run_dir": _portable_path(run_dir),
        "run_meta": meta,
        "summary_path": _portable_path(repo_path(summary_path)) if summary_path else None,
        "evidence_bundle": {
            "path": _portable_path(bundle_dir),
            "files": evidence,
            "binary_artifacts_included": False,
        },
        "gate": {
            "passed": gate_passed,
            "highlights": highlights,
        },
    }
    if extra:
        record.update(extra)

    write_json_atomic(record_path, record)
    append_index(record_path, record)
    update_latest_pointer(stage, record_path)
    if tasks and len(tasks) == 1:
        update_latest_pointer(stage, record_path, task=tasks[0])
    return record_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage")
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    path = emit_stage_record(
        args.stage,
        summary_path=args.summary_path,
        artifact_dir=args.artifact_dir,
        tasks=list(args.tasks),
        label=args.label,
    )
    print(json.dumps({"record": str(path)}, indent=2))
    return 0


def list_records(*, limit: int = 20) -> list[dict[str, Any]]:
    index_path = RECORDS_DIR / "index.jsonl"
    if not index_path.is_file():
        return []
    lines = index_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return rows[-limit:]


def resolve_latest_record(stage: str, *, task: str | None = None) -> Path | None:
    pointer = RECORDS_DIR / (f"LATEST_{stage}_{task}" if task else f"LATEST_{stage}")
    if pointer.is_file():
        candidate = Path(pointer.read_text(encoding="utf-8").strip())
        if candidate.is_file():
            return candidate
    if not RECORDS_DIR.is_dir():
        return None
    pattern = f"*_{stage}"
    if task:
        pattern = f"*_{stage}_{task}"
    matches = sorted(RECORDS_DIR.glob(f"{pattern}.json"), reverse=True)
    return matches[0] if matches else None


if __name__ == "__main__":
    raise SystemExit(main())
