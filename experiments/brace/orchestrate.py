#!/usr/bin/env python3
"""BRACE screen scheduler scaffold (credit / preservation / integration).

Screen training is intentionally gated behind anchor smoke and chunk manifests.
This module wires job planning and resume state; GPU finetune hooks are stubs.
"""

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

from experiments.brace.replay_audit import read_json, write_json_atomic


def load_screen_protocol(path: Path) -> dict[str, Any]:
    return read_json(path)


def discover_chunk_manifests(dataset_dir: Path, run_label: str) -> dict[str, Path | None]:
    b1 = dataset_dir / f"{run_label}_B1.jsonl"
    n1 = dataset_dir / f"{run_label}_N1.jsonl"
    return {
        "B1": b1 if b1.is_file() else None,
        "N1": n1 if n1.is_file() else None,
    }


def build_screen_jobs(
    *,
    protocol: dict[str, Any],
    task: str,
    run_label: str,
    screen: str,
) -> dict[str, Any]:
    dataset_dir = BRACE_DIR / "datasets"
    manifests = discover_chunk_manifests(dataset_dir, run_label)
    epochs = list(protocol.get("screen_epochs", [1, 3, 5, 7, 10]))
    methods = list(protocol["screens"][screen]["methods"])
    jobs: dict[str, Any] = {}
    for method in methods:
        for epoch in epochs:
            job_id = f"{screen}:{task}:{method}:epoch{epoch}"
            data_manifest = manifests.get(method)
            jobs[job_id] = {
                "id": job_id,
                "screen": screen,
                "task": task,
                "method": method,
                "epochs": epoch,
                "status": "pending" if data_manifest is not None or method in {"base", "U1"} else "blocked",
                "manifest": str(data_manifest) if data_manifest is not None else None,
                "command": [
                    "python",
                    str(BRACE_DIR / "chunk_dataset.py"),
                    "--manifest",
                    str(data_manifest) if data_manifest is not None else "",
                ],
                "note": "Wire to DP finetune entry before launching GPU screen.",
            }
    return jobs


def create_state(stage: str, *, task: str, run_label: str, protocol_path: Path) -> dict[str, Any]:
    protocol = load_screen_protocol(protocol_path)
    if stage == "screen":
        jobs: dict[str, Any] = {}
        for screen_name in ("credit", "preservation", "integration"):
            jobs.update(build_screen_jobs(protocol=protocol, task=task, run_label=run_label, screen=screen_name))
        return {
            "stage": stage,
            "task": task,
            "run_label": run_label,
            "protocol_path": str(protocol_path),
            "jobs": jobs,
        }
    if stage == "full":
        raise NotImplementedError("BRACE full eval requires completed screen promotions.")
    raise ValueError(f"Unknown stage: {stage}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["screen", "full"])
    parser.add_argument("--protocol", type=Path, default=BRACE_DIR / "screen_protocol.v1.json")
    parser.add_argument("--task", default="place_container_plate")
    parser.add_argument("--run-label", default="place_pilot_v2.3")
    parser.add_argument("--state", type=Path, default=BRACE_DIR / "run_states" / "screen.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    anchor_summary = BRACE_DIR / "anchor_smoke" / "summary.json"
    if not anchor_summary.is_file() or not read_json(anchor_summary).get("passed"):
        raise SystemExit(f"anchor smoke gate not passed: {anchor_summary}")

    state = create_state(args.stage, task=args.task, run_label=args.run_label, protocol_path=args.protocol)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.state, state)
    print(json.dumps({"job_count": len(state["jobs"]), "state": str(args.state)}, indent=2))
    if args.dry_run:
        return 0
    raise NotImplementedError(
        "BRACE screen GPU training is not wired yet. Use --dry-run to materialize scheduler state."
    )


if __name__ == "__main__":
    raise SystemExit(main())
