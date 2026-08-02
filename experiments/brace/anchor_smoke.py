#!/usr/bin/env python3
"""Aggregate unit and training-path anchor smoke gates for BRACE screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.anchor_unit_smoke import run_unit_anchor_smoke
from experiments.brace.replay_audit import git_commit, read_json, repo_path, write_json_atomic


def anchor_gate_passed(summary: dict[str, Any], protocol_revision: str) -> bool:
    return bool(
        summary.get("passed")
        and summary.get("gate_level") == "training_path"
        and summary.get("protocol_revision") == protocol_revision
        and summary.get("checks", {}).get("training_path_smoke_passed")
        and summary.get("checks", {}).get("unit_smoke_passed")
    )


def run_anchor_smoke(
    protocol: dict[str, Any],
    *,
    task: str | None = None,
    run_label: str | None = None,
    dataset: str = "N1",
    base_checkpoint: Path | None = None,
    traced_root: Path | None = None,
    output_dir: Path | None = None,
    skip_training_path: bool = False,
) -> dict[str, Any]:
    unit_summary = run_unit_anchor_smoke(protocol)
    training_summary: dict[str, Any] | None = None
    if not skip_training_path and task and run_label and base_checkpoint and traced_root:
        from experiments.brace.anchor_training_smoke import run_training_path_smoke

        training_summary = run_training_path_smoke(
            protocol,
            task=task,
            run_label=run_label,
            dataset=dataset,
            traced_root=traced_root,
            base_checkpoint=base_checkpoint,
            work_dir=output_dir / "training_work" if output_dir else None,
        )
    elif not skip_training_path:
        training_summary = {
            "smoke_level": "training_path",
            "eligible_for_screen_gate": True,
            "passed": False,
            "complete": False,
            "error": "missing task/run_label/checkpoint/traced_root for training-path smoke",
        }

    training_passed = bool(training_summary and training_summary.get("passed"))
    unit_passed = bool(unit_summary.get("passed"))
    protocol_revision = protocol.get("protocol_revision", "screen.v1")
    aggregate = {
        "schema_version": 1,
        "protocol_revision": protocol_revision,
        "gate_level": "training_path",
        "eligible_for_screen_gate": True,
        "passed": unit_passed and training_passed,
        "complete": unit_passed and bool(training_summary and training_summary.get("complete")),
        "checks": {
            "unit_smoke_passed": unit_passed,
            "training_path_smoke_passed": training_passed,
        },
        "unit": unit_summary,
        "training_path": training_summary,
        "git_commit": git_commit(),
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output_dir / "unit" / "summary.json", unit_summary)
        if training_summary is not None:
            write_json_atomic(output_dir / "training_path" / "summary.json", training_summary)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("experiments/brace/screen_protocol.v1.1.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/brace/anchor_smoke/summary.json"))
    parser.add_argument("--task", default=None)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--dataset", default="N1")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--traced-rollout-dir", type=Path, default=Path("experiments/brace/rollouts_traced_pilot"))
    parser.add_argument("--structural-only", action="store_true")
    args = parser.parse_args()

    protocol = read_json(repo_path(args.protocol))
    output = repo_path(args.output)
    aggregate = run_anchor_smoke(
        protocol,
        task=args.task,
        run_label=args.run_label,
        dataset=args.dataset,
        base_checkpoint=repo_path(args.checkpoint) if args.checkpoint else None,
        traced_root=repo_path(args.traced_rollout_dir),
        output_dir=output.parent,
        skip_training_path=args.structural_only,
    )
    write_json_atomic(output, aggregate)
    try:
        from experiments.brace.stage_records import emit_stage_record

        emit_stage_record(
            "anchor_smoke",
            summary=aggregate,
            summary_path=output,
            tasks=[args.task] if args.task else None,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to emit stage record: {exc}", file=sys.stderr)
    print(json.dumps(aggregate, indent=2))
    return 0 if aggregate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
