#!/usr/bin/env python3
"""Static audit: detect writes to deprecated mutable BRACE JSON paths."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"

# Paths that must not be write targets (except via promote_run or legacy flag docs).
FORBIDDEN_WRITE_PATTERNS = [
    r"replay_audit_v2/summary\.json",
    r"replay_audit_v2/combined_summary\.json",
    r"branches/summary\.json",
    r"branches/checks\.jsonl",
    r"branches_confirm/merged_gate\.json",
    r"branches_dump/summary\.json",
    r"budgets/.*_pilot\.json",
]

ALLOWLIST_SUBSTRINGS = [
    "promote_run.py",
    "audit_mutable_paths.py",
    "test_track_c_scaffold",
    "archive_place_pilot.sh",
    "archive_dump_pilot.sh",
    "archive_place_confirm.sh",
    "README.md",
    "BRACE_LEGACY_MUTABLE_OUTPUTS",
    "resolve_artifact.py",
    "inventory_artifacts.py",
]

WRITE_HINTS = re.compile(
    r"(write_json(?:l)?_atomic|write_json_atomic|open\([^)]*[\"']w|>\s*\{|cp\s+[\"']|shutil\.copy)",
    re.MULTILINE,
)


def scan_file(path: Path) -> list[str]:
    if path.suffix not in {".py", ".sh"}:
        return []
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(REPO_ROOT).as_posix()
    if any(token in text for token in ALLOWLIST_SUBSTRINGS) and "write_json" not in rel:
        pass
    violations: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if any(token in line for token in ALLOWLIST_SUBSTRINGS):
            continue
        if not WRITE_HINTS.search(line):
            continue
        for pattern in FORBIDDEN_WRITE_PATTERNS:
            if re.search(pattern, line):
                violations.append(f"{rel}:{line_no}: {line.strip()}")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brace-dir", type=Path, default=BRACE_DIR)
    args = parser.parse_args()

    violations: list[str] = []
    for path in sorted(args.brace_dir.rglob("*")):
        if path.is_file():
            violations.extend(scan_file(path))

    if violations:
        print("Mutable path write violations:", file=sys.stderr)
        for item in violations:
            print(item, file=sys.stderr)
        return 1
    print(json.dumps({"passed": True, "violation_count": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
