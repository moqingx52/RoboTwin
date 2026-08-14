#!/usr/bin/env python3
"""Derive feasibility evidence schema v2 from archived schema-v1 evidence.

The original evidence files are kept byte-for-byte untouched. For each
``*_feasibility.json`` without a ``task_status`` field a derived
``*_feasibility_v2.json`` is written that records the original evidence path
and SHA256 under ``provenance.source_evidence``. A schema-v2 batch summary
with repo-relative paths and evidence SHA256 is rebuilt in the same directory.

Usage:
    python experiments/brace/upgrade_archived_feasibility_v2.py \
        --archive-dir experiments/brace/archive/seed_feasibility_pilot_20260807 \
        --tasks beat_block_hammer handover_mic lift_pot open_laptop \
                 place_burger_fries shake_bottle stack_bowls_three
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.aggregate_seed_feasibility_batch import build_batch_summary
from experiments.brace.multitask_protocol import file_sha256
from experiments.brace.replay_audit import read_json, write_json_atomic
from experiments.brace.seed_feasibility import (
    DEFAULT_TASK_CONFIG,
    build_feasibility_evidence,
    derive_task_status,
    validate_feasibility_evidence,
)


def derive_v2_from_legacy(evidence: dict, source: Path) -> dict:
    """Rebuild schema-v2 evidence from a schema-v1 file that lacks task_status."""
    results = evidence.get("results", [])
    candidate_seeds = [int(row["seed"]) for row in results]
    required = evidence.get("expert_demo_count", 50)
    if not candidate_seeds:
        raise ValueError(f"{source}: no probe results to derive from")
    derived = build_feasibility_evidence(
        task=str(evidence.get("task") or source.name),
        candidate_seeds=candidate_seeds,
        probe_results=results,
        task_config=str(evidence.get("task_config", DEFAULT_TASK_CONFIG)),
        candidate_partition=str(evidence.get("candidate_partition", "rollout_train")),
        required_count=int(required),
        rule=str(evidence.get("selection_rule", "first_n_solvable_in_manifest_order")),
        source_evidence={
            "path": str(source.resolve().relative_to(REPO_ROOT.resolve())),
            "sha256": file_sha256(source),
            "schema_version": evidence.get("schema_version", 1),
        },
    )
    for key in ("verify_label",):
        if evidence.get(key) is not None:
            derived[key] = evidence[key]
    return derived


def upgrade_archive(archive_dir: Path, tasks: list[str]) -> dict:
    derived_paths: dict[str, Path] = {}
    for path in sorted(archive_dir.glob("*_feasibility.json")):
        if path.name == "batch_summary.json" or path.name.endswith("_v2.json"):
            continue
        evidence = read_json(path)
        if evidence.get("schema_version") == 2 and evidence.get("task_status"):
            derived_paths[path.name.removesuffix("_feasibility.json")] = path
            continue
        derived = derive_v2_from_legacy(evidence, path)
        errors = validate_feasibility_evidence(derived)
        if errors:
            raise ValueError(f"{path}: derived evidence invalid: {errors}")
        out_path = path.with_name(path.stem + "_v2.json")
        write_json_atomic(out_path, derived)
        derived_paths[path.name.removesuffix("_feasibility.json")] = out_path
    summary = build_batch_summary(archive_dir, tasks)
    summary["archive_dir"] = str(archive_dir.resolve().relative_to(REPO_ROOT.resolve()))
    summary["derived_evidence"] = {
        key: str(path.relative_to(REPO_ROOT.resolve())) for key, path in sorted(derived_paths.items())
    }
    summary["reverify_evidence"] = {}
    for task in tasks:
        reverify = archive_dir / f"{task}_reverify_feasibility_v2.json"
        if reverify.is_file():
            summary["reverify_evidence"][task] = str(reverify.relative_to(REPO_ROOT.resolve()))
    write_json_atomic(archive_dir / "batch_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    args = parser.parse_args()
    summary = upgrade_archive(args.archive_dir.resolve(), args.tasks)
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
