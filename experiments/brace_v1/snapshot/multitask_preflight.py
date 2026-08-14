#!/usr/bin/env python3
"""Fail-closed readiness audit for the BRACE RoboTwin multitask experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
DP_DIR = REPO_ROOT / "policy" / "DP"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.multitask_protocol import file_sha256, resolve_repo_path, validate_multitask_protocol
from experiments.brace.replay_audit import read_json, write_json_atomic


def sidecar_valid(path: Path) -> bool:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        return False
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    return bool(fields) and fields[0] == file_sha256(path)


def replay_gate(task: str) -> tuple[bool, str | None]:
    pointer = BRACE_DIR / "runs" / f"LATEST_AUDIT_{task}"
    candidates: list[Path] = []
    if pointer.is_file():
        candidates.append(Path(pointer.read_text(encoding="utf-8").strip()) / "summary.json")
    candidates.extend((BRACE_DIR / "archive").glob("replay_audit_v2_*/summary.json"))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        payload = read_json(candidate)
        try:
            protocol_version = float(payload.get("protocol_version", payload.get("protocol_revision", 0)))
        except (TypeError, ValueError):
            protocol_version = 0.0
        if protocol_version >= 2.3 and payload.get("tasks", {}).get(task, {}).get("replay_gate_passed") is True:
            return True, str(candidate)
    return False, None


def seed_manifest_gate(seed_manifest: Path) -> tuple[bool, dict[str, Any]]:
    if not seed_manifest.is_file():
        return False, {}
    payload = read_json(seed_manifest)
    feasibility = payload.get("feasibility", {})
    evidence_value = feasibility.get("evidence_path")
    evidence_path = resolve_repo_path(evidence_value) if evidence_value else None
    passed = (
        payload.get("status") == "frozen"
        and feasibility.get("criterion") == "expert_script_simulator_solvability_only"
        and feasibility.get("learning_policy_performance_consulted") is False
        and feasibility.get("passed") is True
        and evidence_path is not None
        and evidence_path.is_file()
        and file_sha256(evidence_path) == feasibility.get("evidence_sha256")
    )
    return passed, payload


def audit_task(task: str, seed_dir: Path) -> dict[str, Any]:
    checkpoint = DP_DIR / "checkpoints" / f"{task}-demo_clean-50-0" / "600.ckpt"
    expert_zarr = DP_DIR / "data" / f"{task}-demo_clean-50.zarr"
    seed_manifest = seed_dir / f"{task}.json"
    replay_passed, replay_summary = replay_gate(task)
    seed_frozen, seed_payload = seed_manifest_gate(seed_manifest)
    checks = {
        "base_checkpoint": checkpoint.is_file(),
        "expert_dataset": expert_zarr.is_dir(),
        "seed_manifest_frozen": seed_frozen,
        "seed_manifest_sha256": sidecar_valid(seed_manifest) if seed_manifest.is_file() else False,
        "replay_gate": replay_passed,
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "paths": {
            "base_checkpoint": str(checkpoint),
            "expert_dataset": str(expert_zarr),
            "seed_manifest": str(seed_manifest),
            "replay_summary": replay_summary,
            "seed_feasibility_evidence": seed_payload.get("feasibility", {}).get("evidence_path"),
        },
        "seed_manifest_sha256": file_sha256(seed_manifest) if seed_manifest.is_file() else None,
    }


def build_preflight(protocol_path: Path, *, require_method_freeze: bool) -> dict[str, Any]:
    validation = validate_multitask_protocol(protocol_path, require_method_freeze=require_method_freeze)
    if not validation["passed"]:
        return {
            "schema_version": 1,
            "stage": "brace_multitask_preflight",
            "ready": False,
            "protocol_validation": validation,
            "tasks": {},
        }
    protocol = read_json(protocol_path.resolve())
    task_manifest = read_json(resolve_repo_path(protocol["task_manifest"]))
    seed_dir = BRACE_DIR / "seeds" / "multitask_v1"
    task_names = task_manifest["development_tasks"] + task_manifest["heldout_tasks"]
    tasks = {task: audit_task(task, seed_dir) for task in task_names}
    heldout = task_manifest["heldout_tasks"]
    return {
        "schema_version": 1,
        "stage": "brace_multitask_preflight",
        "ready": all(tasks[task]["ready"] for task in heldout),
        "protocol_validation": validation,
        "development_tasks": task_manifest["development_tasks"],
        "heldout_tasks": heldout,
        "tasks": tasks,
        "heldout_ready_count": sum(tasks[task]["ready"] for task in heldout),
        "heldout_task_count": len(heldout),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=BRACE_DIR / "multitask_protocol.v1.json")
    parser.add_argument("--require-method-freeze", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = build_preflight(args.protocol, require_method_freeze=args.require_method_freeze)
    if args.output:
        write_json_atomic(args.output, summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
