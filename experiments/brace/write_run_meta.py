#!/usr/bin/env python3
"""Write immutable BRACE run metadata next to timestamped outputs."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.replay_audit import git_commit, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--extra", nargs="*", default=[], help="key=value metadata fields")
    args = parser.parse_args()

    extra: dict[str, Any] = {}
    for item in args.extra:
        key, value = item.split("=", 1)
        extra[key] = value

    payload = {
        "schema_version": 1,
        "run_id": args.run_id,
        "stage": args.stage,
        "tasks": list(args.tasks),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hostname": socket.gethostname(),
        "git_commit": git_commit(),
        "pid": os.getpid(),
        **extra,
    }
    run_dir = args.run_dir if args.run_dir.is_absolute() else REPO_ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(run_dir / "meta.json", payload)
    print(json.dumps({"run_dir": str(run_dir), "meta": str(run_dir / "meta.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
