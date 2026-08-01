#!/usr/bin/env python3
"""Scan BRACE evidence bundles and emit a remote/local sync inventory.

Designed for cloud runs without git push access: scan once, download via checklist.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.replay_audit import file_sha256, git_commit, read_json, repo_path, write_json_atomic


@dataclass(frozen=True)
class ArtifactSpec:
    bundle_id: str
    repo_path: str
    source_paths: tuple[str, ...]
    required: bool = True
    git_track: bool = True


BUNDLES: tuple[ArtifactSpec, ...] = (
    ArtifactSpec(
        "place_pilot_archive",
        "experiments/brace/archive/branches_place_pilot_valid_v2.3/checks.jsonl",
        (
            "experiments/brace/archive/branches_place_pilot_valid_v2.3/checks.jsonl",
            "experiments/brace/branches/checks.jsonl",
        ),
    ),
    ArtifactSpec(
        "place_pilot_archive",
        "experiments/brace/archive/branches_place_pilot_valid_v2.3/summary.json",
        (
            "experiments/brace/archive/branches_place_pilot_valid_v2.3/summary.json",
            "experiments/brace/branches/summary.json",
        ),
    ),
    ArtifactSpec(
        "place_pilot_archive",
        "experiments/brace/archive/branches_place_pilot_valid_v2.3/protocol.v2.3.json",
        (
            "experiments/brace/archive/branches_place_pilot_valid_v2.3/protocol.v2.3.json",
            "experiments/brace/protocol.v2.3.json",
        ),
    ),
    ArtifactSpec(
        "place_pilot_archive",
        "experiments/brace/archive/branches_place_pilot_valid_v2.3/analyzed_seeds.json",
        ("experiments/brace/archive/branches_place_pilot_valid_v2.3/analyzed_seeds.json",),
        required=False,
    ),
    ArtifactSpec(
        "place_pilot_archive",
        "experiments/brace/archive/branches_place_pilot_valid_v2.3/MANIFEST.sha256",
        ("experiments/brace/archive/branches_place_pilot_valid_v2.3/MANIFEST.sha256",),
        required=False,
    ),
    ArtifactSpec(
        "dump_pilot_archive",
        "experiments/brace/archive/branches_dump_pilot_valid_v2.3/checks.jsonl",
        (
            "experiments/brace/archive/branches_dump_pilot_valid_v2.3/checks.jsonl",
            "experiments/brace/branches_dump/checks.jsonl",
        ),
    ),
    ArtifactSpec(
        "dump_pilot_archive",
        "experiments/brace/archive/branches_dump_pilot_valid_v2.3/summary.json",
        (
            "experiments/brace/archive/branches_dump_pilot_valid_v2.3/summary.json",
            "experiments/brace/branches_dump/summary.json",
        ),
    ),
    ArtifactSpec(
        "dump_pilot_archive",
        "experiments/brace/archive/branches_dump_pilot_valid_v2.3/protocol.v2.3.json",
        (
            "experiments/brace/archive/branches_dump_pilot_valid_v2.3/protocol.v2.3.json",
            "experiments/brace/protocol.v2.3.json",
        ),
    ),
    ArtifactSpec(
        "dump_pilot_archive",
        "experiments/brace/archive/branches_dump_pilot_valid_v2.3/analyzed_seeds.json",
        ("experiments/brace/archive/branches_dump_pilot_valid_v2.3/analyzed_seeds.json",),
        required=False,
    ),
    ArtifactSpec(
        "dump_pilot_archive",
        "experiments/brace/archive/branches_dump_pilot_valid_v2.3/MANIFEST.sha256",
        ("experiments/brace/archive/branches_dump_pilot_valid_v2.3/MANIFEST.sha256",),
        required=False,
    ),
    ArtifactSpec(
        "confirm_archive",
        "experiments/brace/archive/branches_place_confirm_v2.3/summary.json",
        (
            "experiments/brace/archive/branches_place_confirm_v2.3/summary.json",
            "experiments/brace/branches_confirm/summary.json",
        ),
    ),
    ArtifactSpec(
        "confirm_archive",
        "experiments/brace/archive/branches_place_confirm_v2.3/checks.jsonl",
        (
            "experiments/brace/archive/branches_place_confirm_v2.3/checks.jsonl",
            "experiments/brace/branches_confirm/checks.jsonl",
        ),
    ),
    ArtifactSpec(
        "confirm_archive",
        "experiments/brace/archive/branches_place_confirm_v2.3/merged_gate.json",
        (
            "experiments/brace/archive/branches_place_confirm_v2.3/merged_gate.json",
            "experiments/brace/branches_confirm/merged_gate.json",
        ),
    ),
    ArtifactSpec(
        "confirm_archive",
        "experiments/brace/archive/branches_place_confirm_v2.3/protocol.v2.3.json",
        (
            "experiments/brace/archive/branches_place_confirm_v2.3/protocol.v2.3.json",
            "experiments/brace/protocol.v2.3.json",
        ),
        required=False,
    ),
    ArtifactSpec(
        "datasets_b1n1",
        "experiments/brace/datasets/place_pilot_v2.3_B1.jsonl",
        ("experiments/brace/datasets/place_pilot_v2.3_B1.jsonl",),
    ),
    ArtifactSpec(
        "datasets_b1n1",
        "experiments/brace/datasets/place_pilot_v2.3_N1.jsonl",
        ("experiments/brace/datasets/place_pilot_v2.3_N1.jsonl",),
    ),
    ArtifactSpec(
        "datasets_b1n1",
        "experiments/brace/datasets/export_summary.json",
        ("experiments/brace/datasets/export_summary.json",),
        required=False,
    ),
    ArtifactSpec(
        "replay_gate_place",
        "experiments/brace/archive/replay_audit_v2_place_v2.3_gate/summary.json",
        (
            "experiments/brace/archive/replay_audit_v2_place_v2.3_gate/summary.json",
            "experiments/brace/replay_audit_v2/place_container_plate/summary.json",
        ),
    ),
    ArtifactSpec(
        "replay_gate_dump",
        "experiments/brace/archive/replay_audit_v2_dump_v2.3_gate/summary.json",
        (
            "experiments/brace/archive/replay_audit_v2_dump_v2.3_gate/summary.json",
            "experiments/brace/replay_audit_v2/dump_bin_bigbin/summary.json",
        ),
    ),
    ArtifactSpec(
        "anchor_smoke",
        "experiments/brace/anchor_smoke/summary.json",
        ("experiments/brace/anchor_smoke/summary.json",),
    ),
    ArtifactSpec(
        "seeds_confirm",
        "experiments/brace/seeds/place_container_plate_confirm_seeds.json",
        ("experiments/brace/seeds/place_container_plate_confirm_seeds.json",),
        required=False,
    ),
    ArtifactSpec(
        "working_copy_place",
        "experiments/brace/branches/checks.jsonl",
        ("experiments/brace/branches/checks.jsonl",),
        required=False,
        git_track=False,
    ),
    ArtifactSpec(
        "working_copy_confirm",
        "experiments/brace/branches_confirm/checks.jsonl",
        ("experiments/brace/branches_confirm/checks.jsonl",),
        required=False,
        git_track=False,
    ),
    ArtifactSpec(
        "working_copy_dump",
        "experiments/brace/branches_dump/checks.jsonl",
        ("experiments/brace/branches_dump/checks.jsonl",),
        required=False,
        git_track=False,
    ),
)


def count_jsonl_lines(path: Path) -> int | None:
    if not path.is_file():
        return None
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def resolve_source(spec: ArtifactSpec) -> tuple[Path | None, str | None]:
    for candidate in spec.source_paths:
        path = repo_path(candidate)
        if path.is_file():
            return path, candidate
    return None, None


def inspect_artifact(spec: ArtifactSpec) -> dict[str, Any]:
    source_path, source_rel = resolve_source(spec)
    repo_target = repo_path(spec.repo_path)
    entry: dict[str, Any] = {
        "bundle_id": spec.bundle_id,
        "repo_path": spec.repo_path,
        "required": spec.required,
        "git_track": spec.git_track,
        "source_path": source_rel,
        "exists": source_path is not None,
        "status": "present" if source_path is not None else "missing",
    }
    if source_path is None:
        return entry

    entry["size_bytes"] = source_path.stat().st_size
    entry["sha256"] = file_sha256(source_path)
    if source_path.suffix == ".jsonl":
        entry["line_count"] = count_jsonl_lines(source_path)

    if repo_target.is_file() and repo_target != source_path:
        local_hash = file_sha256(repo_target)
        entry["local_sha256"] = local_hash
        if local_hash != entry["sha256"]:
            entry["status"] = "stale"
    return entry


def bundle_summary(entries: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for entry in entries:
        bundle = summary.setdefault(
            entry["bundle_id"],
            {"required": 0, "present": 0, "missing": 0, "git_track": 0},
        )
        if entry["required"]:
            bundle["required"] += 1
        if entry.get("git_track"):
            bundle["git_track"] += 1
        if entry["exists"]:
            bundle["present"] += 1
        elif entry["required"]:
            bundle["missing"] += 1
    return summary


def build_inventory() -> dict[str, Any]:
    entries = [inspect_artifact(spec) for spec in BUNDLES]
    protocol_path = BRACE_DIR / "protocol.v2.3.json"
    protocol_revision = None
    if protocol_path.is_file():
        protocol_revision = read_json(protocol_path).get("protocol_revision")
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hostname": socket.gethostname(),
        "git_commit": git_commit(),
        "protocol_revision": protocol_revision,
        "bundle_summary": bundle_summary(entries),
        "artifacts": entries,
    }


def render_checklist(inventory: dict[str, Any], *, remote_repo: str) -> str:
    lines = [
        "# BRACE artifact sync checklist",
        "",
        f"Generated: {inventory['generated_at']} UTC",
        f"Host: {inventory['hostname']}",
        f"Git commit: {inventory.get('git_commit') or 'unknown'}",
        f"Protocol: v{inventory.get('protocol_revision') or '?'}",
        "",
        "## Bundle summary",
        "",
        "| Bundle | Required | Present | Missing | Git-track items |",
        "|--------|----------|---------|---------|-----------------|",
    ]
    for bundle_id, stats in sorted(inventory["bundle_summary"].items()):
        lines.append(
            f"| {bundle_id} | {stats['required']} | {stats['present']} | {stats['missing']} | {stats['git_track']} |"
        )

    lines.extend(
        [
            "",
            "## Download to local repo paths",
            "",
            "Replace `REMOTE_HOST` and `REMOTE_REPO` before running.",
            "",
            "| Status | Repo path | Source | SHA256 | Size |",
            "|--------|-----------|--------|--------|------|",
        ]
    )
    for entry in inventory["artifacts"]:
        if not entry.get("git_track") or not entry.get("exists"):
            continue
        size = entry.get("size_bytes", 0)
        lines.append(
            f"| {entry['status']} | `{entry['repo_path']}` | `{entry['source_path']}` | "
            f"`{entry.get('sha256', '')[:16]}…` | {size} |"
        )

    lines.extend(["", "## Copy commands", ""])
    for entry in inventory["artifacts"]:
        if not entry.get("git_track") or not entry.get("exists"):
            continue
        dest = entry["repo_path"]
        src = entry["source_path"]
        lines.append("```bash")
        lines.append(f"mkdir -p \"$(dirname {dest})\"")
        lines.append(
            f"scp REMOTE_HOST:{remote_repo}/{src} {dest}"
        )
        lines.append("```")
        lines.append("")

    lines.extend(
        [
            "## After download (local machine)",
            "",
            "```bash",
            "bash experiments/brace/archive_place_pilot.sh",
            "bash experiments/brace/archive_dump_pilot.sh",
            "bash experiments/brace/archive_place_confirm.sh  # if confirm artifacts present",
            "python experiments/brace/validate_artifacts.py --inventory experiments/brace/sync/LATEST",
            "```",
            "",
            "HDF5 under `experiments/brace/rollouts_traced*` is **not** tracked in git.",
            "Manifests may reference HDF5 paths that must exist on the training host.",
            "",
        ]
    )
    return "\n".join(lines)


def write_inventory_outputs(
    inventory: dict[str, Any],
    *,
    inventory_dir: Path,
    checklist_path: Path,
    remote_repo: str,
) -> Path:
    inventory_dir.mkdir(parents=True, exist_ok=True)
    timestamp = inventory["generated_at"].replace(":", "").replace("-", "")
    inventory_path = inventory_dir / f"{timestamp}.json"
    write_json_atomic(inventory_path, inventory)
    latest_pointer = inventory_dir.parent / "LATEST"
    latest_pointer.write_text(f"inventories/{inventory_path.name}\n", encoding="utf-8")
    checklist_path.parent.mkdir(parents=True, exist_ok=True)
    checklist_path.write_text(render_checklist(inventory, remote_repo=remote_repo), encoding="utf-8")
    return inventory_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory-dir",
        type=Path,
        default=BRACE_DIR / "sync" / "inventories",
    )
    parser.add_argument(
        "--checklist",
        type=Path,
        default=BRACE_DIR / "sync" / "CHECKLIST.md",
    )
    parser.add_argument(
        "--remote-repo",
        default="/workspace/RoboTwin",
        help="Remote absolute repo root for scp examples.",
    )
    args = parser.parse_args()

    inventory = build_inventory()
    inventory_path = write_inventory_outputs(
        inventory,
        inventory_dir=repo_path(args.inventory_dir),
        checklist_path=repo_path(args.checklist),
        remote_repo=args.remote_repo,
    )
    missing_required = [
        entry["repo_path"]
        for entry in inventory["artifacts"]
        if entry["required"] and not entry["exists"]
    ]
    print(json.dumps({"inventory": str(inventory_path), "missing_required": missing_required}, indent=2))
    return 1 if missing_required else 0


if __name__ == "__main__":
    raise SystemExit(main())
