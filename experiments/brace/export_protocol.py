#!/usr/bin/env python3
"""Export protocol for verified-chunk manifests with strict credit matching."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_export_protocol(path: Path | None = None) -> dict:
    protocol_path = path or REPO_ROOT / "experiments/brace/export_protocol.v1.2.json"
    return json.loads(protocol_path.read_text(encoding="utf-8"))
