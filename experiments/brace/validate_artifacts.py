#!/usr/bin/env python3
"""Validate BRACE evidence bundles against manifests and optional remote inventory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.replay_audit import file_sha256, read_json, repo_path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_inventory(path: Path) -> dict[str, Any]:
    if path.name == "LATEST":
        pointer = path.read_text(encoding="utf-8").strip()
        candidates = [
            path.parent / pointer,
            path.parent / "inventories" / Path(pointer).name,
        ]
        for candidate in candidates:
            if candidate.is_file():
                path = candidate
                break
        else:
            raise FileNotFoundError(
                f"inventory pointer not found: {pointer} (tried {', '.join(str(c) for c in candidates)})"
            )
    return read_json(path)


def validate_replay_gate_archive(summary_path: Path, errors: list[str]) -> None:
    if not summary_path.is_file():
        return
    summary = read_json(summary_path)
    replay = summary.get("control_trace_replay", {})
    if int(replay.get("total_checks", 0)) == 0:
        errors.append(f"replay gate archive is empty shell: {summary_path}")
    if not summary.get("complete"):
        errors.append(f"replay gate archive incomplete: {summary_path}")


def validate_source_run(archive_dir: Path, errors: list[str], warnings: list[str]) -> None:
    source_run = archive_dir / "source_run.json"
    if not source_run.is_file():
        warnings.append(f"missing source_run.json (pre-promote-run archive): {archive_dir}")
        return
    payload = read_json(source_run)
    if not payload.get("source_run_dir") and not payload.get("source"):
        errors.append(f"source_run.json missing source lineage: {source_run}")


def validate_manifest(archive_dir: Path, errors: list[str], warnings: list[str]) -> None:
    manifest_path = archive_dir / "MANIFEST.sha256"
    if not manifest_path.is_file():
        warnings.append(f"missing MANIFEST: {manifest_path}")
        return
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, rel_name = line.split(maxsplit=1)
        rel_name = rel_name.strip()
        if rel_name.startswith(archive_dir.as_posix() + "/"):
            file_path = Path(rel_name)
        else:
            file_path = archive_dir / Path(rel_name).name
        if not file_path.is_file():
            errors.append(f"MANIFEST entry missing file: {file_path}")
            continue
        if file_sha256(file_path) != digest:
            errors.append(f"MANIFEST hash mismatch: {file_path}")


def validate_branch_summary(summary_path: Path, checks_path: Path, errors: list[str]) -> None:
    if not summary_path.is_file():
        errors.append(f"missing summary: {summary_path}")
        return
    summary = read_json(summary_path)
    if "harness_valid" not in summary:
        errors.append(f"summary missing harness_valid: {summary_path}")
    for task_stats in summary.get("tasks", {}).values():
        if "points" not in task_stats:
            errors.append(
                f"summary missing per-point evidence (recover full cloud summary or rebuild from checks): "
                f"{summary_path}"
            )
    if not checks_path.is_file():
        errors.append(f"missing checks.jsonl: {checks_path}")
        return
    checks = read_jsonl(checks_path)
    for task_stats in summary.get("tasks", {}).values():
        expected = int(task_stats.get("total_points", 0))
        if expected and len(checks) < expected:
            errors.append(
                f"checks.jsonl rows ({len(checks)}) < summary total_points ({expected}) for {summary_path}"
            )
            break


def validate_datasets(dataset_dir: Path, run_label: str, errors: list[str], warnings: list[str]) -> None:
    b1 = dataset_dir / f"{run_label}_B1.jsonl"
    n1 = dataset_dir / f"{run_label}_N1.jsonl"
    if not b1.is_file() or not n1.is_file():
        errors.append(f"missing B1/N1 manifests under {dataset_dir} for {run_label}")
        return
    b1_rows = read_jsonl(b1)
    n1_rows = read_jsonl(n1)
    if len(b1_rows) != len(n1_rows):
        errors.append(f"B1/N1 row count mismatch: {len(b1_rows)} vs {len(n1_rows)}")
    for label, rows in (("B1", b1_rows), ("N1", n1_rows)):
        for row in rows:
            if row.get("dataset") != label:
                errors.append(f"{label} manifest has unexpected dataset field: {row.get('dataset')}")
            hdf5 = row.get("hdf5_path")
            if hdf5 and not repo_path(hdf5).is_file():
                warnings.append(f"{label} hdf5_path not found locally (expected off-repo): {hdf5}")


def validate_inventory(inventory: dict[str, Any], *, missing_only: bool, errors: list[str]) -> None:
    by_repo = {entry["repo_path"]: entry for entry in inventory.get("artifacts", [])}
    for repo_rel, entry in by_repo.items():
        if not entry.get("git_track"):
            continue
        if not entry.get("exists"):
            if entry.get("required"):
                errors.append(f"inventory marks required artifact missing on remote: {repo_rel}")
            continue
        path = repo_path(repo_rel)
        if not path.is_file():
            if missing_only or entry.get("required"):
                errors.append(f"local missing (inventory expected present): {repo_rel}")
            continue
        expected = entry.get("sha256")
        if expected and file_sha256(path) != expected:
            errors.append(f"local hash mismatch vs inventory: {repo_rel}")


def run_validation(
    *,
    inventory_path: Path | None,
    missing_only: bool,
    strict_json: bool,
    run_label: str,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if inventory_path is not None:
        inventory = load_inventory(inventory_path)
        validate_inventory(inventory, missing_only=missing_only, errors=errors)

    place_archive = BRACE_DIR / "archive" / "branches_place_pilot_valid_v2.3"
    dump_archive = BRACE_DIR / "archive" / "branches_dump_pilot_valid_v2.3"
    confirm_archive = BRACE_DIR / "archive" / "branches_place_confirm_v2.3"
    place_replay_archive = BRACE_DIR / "archive" / "replay_audit_v2_place_v2.3_gate"
    dump_replay_archive = BRACE_DIR / "archive" / "replay_audit_v2_dump_v2.3_gate"

    for archive in (place_archive, dump_archive, confirm_archive):
        if archive.is_dir():
            validate_source_run(archive, errors, warnings)
    for replay_archive in (place_replay_archive, dump_replay_archive):
        if (replay_archive / "summary.json").is_file():
            validate_replay_gate_archive(replay_archive / "summary.json", errors)
            validate_source_run(replay_archive, errors, warnings)

    if place_archive.is_dir():
        validate_manifest(place_archive, errors, warnings)
        validate_branch_summary(place_archive / "summary.json", place_archive / "checks.jsonl", errors)
        manifest = place_archive / "MANIFEST.sha256"
        if manifest.is_file() and "checks.jsonl" not in manifest.read_text(encoding="utf-8"):
            errors.append("place pilot MANIFEST must include checks.jsonl")

    if dump_archive.is_dir():
        validate_manifest(dump_archive, errors, warnings)
        validate_branch_summary(dump_archive / "summary.json", dump_archive / "checks.jsonl", errors)

    if (confirm_archive / "summary.json").is_file():
        validate_branch_summary(
            confirm_archive / "summary.json",
            confirm_archive / "checks.jsonl",
            errors,
        )
        if not (confirm_archive / "merged_gate.json").is_file():
            errors.append(f"missing merged_gate.json in {confirm_archive}")

    dataset_dir = BRACE_DIR / "datasets"
    if dataset_dir.is_dir():
        validate_datasets(dataset_dir, run_label, errors, warnings)

    passed = not errors and (not strict_json or not warnings)
    return {
        "passed": passed,
        "errors": errors,
        "warnings": warnings,
        "error_count": len(errors),
        "warning_count": len(warnings),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=None)
    parser.add_argument("--missing-only", action="store_true")
    parser.add_argument("--strict-json", action="store_true")
    parser.add_argument("--run-label", default="place_pilot_v2.3")
    args = parser.parse_args()

    inventory_path = repo_path(args.inventory) if args.inventory else None
    if inventory_path is not None and not inventory_path.exists():
        print(json.dumps({"passed": False, "errors": [f"inventory not found: {inventory_path}"]}, indent=2))
        return 2

    result = run_validation(
        inventory_path=inventory_path,
        missing_only=args.missing_only,
        strict_json=args.strict_json,
        run_label=args.run_label,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
