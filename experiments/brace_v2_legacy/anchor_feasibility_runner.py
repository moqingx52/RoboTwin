#!/usr/bin/env python3
"""Short GPU diagnostic that runs pre-registered anchor feasibility optimizer steps."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace_v2_legacy.anchor_feasibility_diagnostic import run_feasibility_diagnostic
from experiments.brace.build_screen_dataset import sha256 as file_sha256
from experiments.brace.replay_audit import git_commit, read_json, repo_path, write_json_atomic


def run_anchor_feasibility(
    protocol: dict,
    *,
    task: str,
    run_label: str,
    dataset: str,
    traced_root: Path,
    base_checkpoint: Path,
    work_dir: Path | None = None,
) -> dict:
    if not torch.cuda.is_available():
        return {
            "schema_version": 3,
            "stage": "anchor_feasibility",
            "passed": False,
            "complete": False,
            "error": "CUDA is required for anchor feasibility diagnostic",
            "git_commit": git_commit(),
        }

    cleanup = None
    if work_dir is None:
        cleanup = tempfile.TemporaryDirectory()
        work_dir = Path(cleanup.name)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        return run_feasibility_diagnostic(
            protocol,
            task=task,
            run_label=run_label,
            dataset=dataset,
            traced_root=traced_root,
            base_checkpoint=base_checkpoint,
            work_dir=work_dir,
        )
    finally:
        if cleanup is not None:
            cleanup.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("experiments/brace/screen_protocol.v1.2.json"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--dataset", default="N1")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--traced-rollout-dir", type=Path, default=BRACE_DIR / "rollouts_traced")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, help="Optional persistent directory for trajectory artifacts")
    args = parser.parse_args()
    protocol_path = repo_path(args.protocol)
    protocol = read_json(protocol_path)
    output = repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    work_dir = repo_path(args.work_dir) if args.work_dir else output.parent
    summary = run_anchor_feasibility(
        protocol,
        task=args.task,
        run_label=args.run_label,
        dataset=args.dataset,
        traced_root=repo_path(args.traced_rollout_dir),
        base_checkpoint=repo_path(args.checkpoint),
        work_dir=work_dir,
    )
    summary["protocol_path"] = str(protocol_path)
    summary["protocol_sha256"] = file_sha256(protocol_path)
    write_json_atomic(output, summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
