#!/usr/bin/env python3
"""Evaluate BRACE anchor constraint feasibility from training logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.build_screen_dataset import sha256 as file_sha256
from experiments.brace.replay_audit import read_json, repo_path, write_json_atomic
from experiments.brace.screen_gates import evaluate_constraint_feasibility, parse_training_logs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("experiments/brace/screen_protocol.v1.2.json"))
    parser.add_argument("--log", type=Path, required=True, help="Hydra logs.json.txt from anchor training")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol_path = repo_path(args.protocol)
    protocol = read_json(protocol_path)
    rows = parse_training_logs(repo_path(args.log))
    if not rows:
        summary = {
            "passed": False,
            "missing_groups": list(evaluate_constraint_feasibility([], protocol)["missing_groups"]),
            "groups": {},
            "error": "empty_training_log",
            "protocol_path": str(protocol_path),
            "protocol_sha256": file_sha256(protocol_path),
        }
        if args.output:
            write_json_atomic(repo_path(args.output), summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1
    summary = evaluate_constraint_feasibility(rows, protocol)
    summary["protocol_path"] = str(protocol_path)
    summary["protocol_sha256"] = file_sha256(protocol_path)
    if args.output:
        write_json_atomic(repo_path(args.output), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        return 1
    if summary.get("missing_groups"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
