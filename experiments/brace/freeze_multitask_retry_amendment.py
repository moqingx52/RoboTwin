#!/usr/bin/env python3
"""Freeze bounded fixed-seed pre-motion retry from completed parity reports."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
PROTOCOL_PATH = BRACE_DIR / "multitask_protocol.v1.json"
V11_PATH = BRACE_DIR / "multitask_amendment.v1.1.json"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.multitask_protocol import file_sha256
from experiments.brace.premotion_retry import validate_retry_amendment
from experiments.brace.replay_audit import write_json_atomic


ELIGIBLE_TASKS = [
    "beat_block_hammer",
    "click_alarmclock",
    "lift_pot",
    "move_can_pot",
    "open_laptop",
    "place_burger_fries",
    "shake_bottle",
    "stack_bowls_three",
]


def validate_parity_report(report: dict[str, Any], *, condition: str) -> list[str]:
    errors: list[str] = []
    if report.get("stage") != "line_b_collect_path_parity_diagnostic":
        errors.append(f"{condition}: unexpected stage")
    if report.get("diagnostic_only") is not True:
        errors.append(f"{condition}: report is not marked diagnostic_only")
    if report.get("condition") != condition:
        errors.append(f"{condition}: condition mismatch")
    if report.get("worker_errors") != []:
        errors.append(f"{condition}: worker_errors is not empty")
    if int(report.get("repeats", 0)) < 5:
        errors.append(f"{condition}: fewer than 5 repeats")
    if not report.get("rows"):
        errors.append(f"{condition}: missing attempt rows")
    return errors


def build_amendment(
    isolated: dict[str, Any],
    packed: dict[str, Any],
    *,
    isolated_archive_path: Path,
    packed_archive_path: Path,
    max_attempts: int,
) -> dict[str, Any]:
    errors = validate_parity_report(isolated, condition="isolated")
    errors.extend(validate_parity_report(packed, condition="packed"))
    if isolated.get("git_commit") != packed.get("git_commit"):
        errors.append("parity reports were not produced from the same git commit")
    if errors:
        raise ValueError("; ".join(errors))
    payload = {
        "schema_version": 1,
        "amendment_revision": "brace.multitask.v1.2",
        "status": "frozen",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "applies_to": {
            "protocol_revision": "brace.multitask.v1",
            "protocol_sha256": file_sha256(PROTOCOL_PATH),
            "prior_amendment": str(V11_PATH.relative_to(REPO_ROOT)),
            "prior_amendment_sha256": file_sha256(V11_PATH),
        },
        "evidence": {
            "diagnostic_git_commit": isolated.get("git_commit"),
            "isolated_parity_report": str(isolated_archive_path.relative_to(REPO_ROOT)),
            "isolated_parity_sha256": file_sha256(isolated_archive_path),
            "packed_parity_report": str(packed_archive_path.relative_to(REPO_ROOT)),
            "packed_parity_sha256": file_sha256(packed_archive_path),
        },
        "operational_collection_retry": {
            "scope": "frozen_expert_demo_saved_seed_premotion_materialization",
            "eligible_tasks": ELIGIBLE_TASKS,
            "max_attempts_per_episode": int(max_attempts),
            "seed_substitution": False,
            "episode_index_rule": "frozen_cohort_index",
            "acceptance_rule": "plan_success_and_check_success_and_trajectory_saved",
            "log_all_attempts": True,
            "exhaustion_rule": "fail_task_without_replacing_or_dropping_seed",
            "existing_first_attempt_successes": "eligible_if_original_collect_log_and_cohort_provenance_are_preserved",
        },
        "decision": [
            "GPU packing was not the primary failure cause because packed outcomes were not systematically worse than isolated outcomes.",
            "Bounded retry applies to the same frozen seed and real cohort episode index; it never substitutes a seed.",
            "handover_mic is excluded pending expert/check repair and a fresh feasibility scan.",
            "put_object_cabinet remains a protocol-feasibility failure and is not eligible.",
        ],
    }
    amendment_errors = validate_retry_amendment(payload, max_attempts=max_attempts)
    if amendment_errors:
        raise ValueError("invalid generated amendment: " + "; ".join(amendment_errors))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolated-report", type=Path, required=True)
    parser.add_argument("--packed-report", type=Path, required=True)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=BRACE_DIR / "archive" / "line_b_collect_parity_20260809",
    )
    parser.add_argument("--output", type=Path, default=BRACE_DIR / "multitask_amendment.v1.2.json")
    parser.add_argument("--max-attempts", type=int, default=5)
    args = parser.parse_args()
    if args.max_attempts < 2:
        raise SystemExit("max-attempts must be at least 2 for a retry amendment")
    isolated_source = args.isolated_report.resolve()
    packed_source = args.packed_report.resolve()
    isolated_source_payload = json.loads(isolated_source.read_text(encoding="utf-8"))
    packed_source_payload = json.loads(packed_source.read_text(encoding="utf-8"))
    source_errors = validate_parity_report(isolated_source_payload, condition="isolated")
    source_errors.extend(validate_parity_report(packed_source_payload, condition="packed"))
    if isolated_source_payload.get("git_commit") != packed_source_payload.get("git_commit"):
        source_errors.append("parity reports were not produced from the same git commit")
    if source_errors:
        raise SystemExit("invalid parity evidence: " + "; ".join(source_errors))
    output = args.output.resolve()
    archive_dir = args.archive_dir.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing amendment: {output}")
    if archive_dir.exists():
        raise SystemExit(f"refusing to overwrite existing archive directory: {archive_dir}")
    archive_dir.mkdir(parents=True)
    isolated_archive = archive_dir / "isolated_report.json"
    packed_archive = archive_dir / "packed_report.json"
    shutil.copyfile(isolated_source, isolated_archive)
    shutil.copyfile(packed_source, packed_archive)
    isolated = json.loads(isolated_archive.read_text(encoding="utf-8"))
    packed = json.loads(packed_archive.read_text(encoding="utf-8"))
    payload = build_amendment(
        isolated,
        packed,
        isolated_archive_path=isolated_archive,
        packed_archive_path=packed_archive,
        max_attempts=args.max_attempts,
    )
    write_json_atomic(output, payload)
    digest = file_sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": digest, "archive_dir": str(archive_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
