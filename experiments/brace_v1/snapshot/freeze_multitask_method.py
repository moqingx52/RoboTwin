#!/usr/bin/env python3
"""Create the immutable BRACE-v2 method freeze after place development."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.multitask_protocol import file_sha256, validate_method_pilot_summary
from experiments.brace.replay_audit import read_json, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--method-config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=BRACE_DIR / "multitask_protocol.v1.json")
    parser.add_argument("--output", type=Path, default=BRACE_DIR / "multitask_method_freeze.v1.json")
    args = parser.parse_args()
    source_run = args.source_run.resolve()
    source_summary = source_run / "summary.json"
    config_path = args.method_config.resolve()
    if not source_summary.is_file() or not config_path.is_file():
        raise SystemExit("source run must contain summary.json and method config must exist")
    summary = read_json(source_summary)
    protocol = read_json(args.protocol.resolve())
    pilot_errors = validate_method_pilot_summary(summary, protocol.get("method_development_gate", {}))
    if pilot_errors:
        raise SystemExit("source development summary failed freeze gate: " + "; ".join(pilot_errors))
    config = read_json(config_path)
    required = {"anchor_construction", "branch_acceptance", "training", "checkpoint_selection"}
    missing = sorted(required - set(config))
    if missing:
        raise SystemExit(f"method config is missing required sections: {missing}")
    payload = {
        "schema_version": 1,
        "freeze_id": "brace_v2_method.v1",
        "status": "frozen",
        "frozen": True,
        "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "development_task": "place_container_plate",
        "source_run": str(source_run),
        "source_summary_sha256": file_sha256(source_summary),
        "source_pilot_gate": summary["preservation_go_no_go"],
        "method_config_path": str(config_path),
        "method_config_sha256": file_sha256(config_path),
        **{key: config[key] for key in sorted(required)},
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
