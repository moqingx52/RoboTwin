#!/usr/bin/env python3
"""Scan rollout_train candidates and materialize expert_demo feasibility evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.multitask_protocol import file_sha256
from experiments.brace.replay_audit import read_json, write_json_atomic
from experiments.brace.seed_feasibility import (
    DEFAULT_CANDIDATE_PARTITION,
    DEFAULT_EXPERT_DEMO_COUNT,
    DEFAULT_TASK_CONFIG,
    TASK_STATUS_PASSED,
    apply_provisional_expert_demo_to_manifest,
    build_feasibility_evidence,
    scan_candidate_seeds,
)


def shard_slice(length: int, shard_id: int, num_shards: int) -> tuple[int, int]:
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("shard_id out of range")
    base, remainder = divmod(length, num_shards)
    start = shard_id * base + min(shard_id, remainder)
    stop = start + base + (1 if shard_id < remainder else 0)
    return start, stop


def merge_probe_results(paths: list[Path]) -> list[dict]:
    merged: list[dict] = []
    seen: set[int] = set()
    for path in paths:
        payload = read_json(path)
        for row in payload.get("results", []):
            seed = int(row["seed"])
            if seed in seen:
                continue
            seen.add(seed)
            merged.append(row)
    return merged


def write_provisional_manifest_update(
    manifest_path: Path,
    manifest: dict,
    evidence: dict[str, object],
    output: Path,
) -> None:
    updated = apply_provisional_expert_demo_to_manifest(manifest, evidence)
    rel_evidence = output.resolve().relative_to(REPO_ROOT.resolve())
    updated["expert_demo_selection"]["evidence_path"] = str(rel_evidence)
    updated["expert_demo_selection"]["evidence_sha256"] = file_sha256(output)
    updated["feasibility"]["evidence_path"] = str(rel_evidence)
    updated["feasibility"]["evidence_sha256"] = file_sha256(output)
    write_json_atomic(manifest_path, updated)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--manifest-dir", type=Path, default=BRACE_DIR / "seeds" / "multitask_v1")
    parser.add_argument("--candidate-partition", default=DEFAULT_CANDIDATE_PARTITION)
    parser.add_argument("--task-config", default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--required-count", type=int, default=DEFAULT_EXPERT_DEMO_COUNT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--merge-inputs", nargs="*")
    parser.add_argument("--provisional-manifest-update", action="store_true")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--verify-label", default="")
    args = parser.parse_args()

    manifest_path = args.manifest_dir / f"{args.task}.json"
    manifest = read_json(manifest_path)
    candidates = manifest.get("partitions", {}).get(args.candidate_partition, [])
    if not candidates:
        raise SystemExit(f"missing candidate partition {args.candidate_partition} in {manifest_path}")

    if args.merge_inputs:
        probe_results = merge_probe_results([Path(path) for path in args.merge_inputs])
    else:
        start, stop = shard_slice(len(candidates), args.shard_id, args.num_shards)
        probe_results = scan_candidate_seeds(
            args.task,
            candidates,
            task_config=args.task_config,
            start_index=start,
            limit=stop - start,
        )
        if args.num_shards > 1:
            label = args.verify_label or args.task
            shard_output = args.output or (
                BRACE_DIR / "runs" / f"seed_feasibility_{label}_shard{args.shard_id:02d}.json"
            )
            shard_payload = {
                "task": args.task,
                "verify_label": args.verify_label or None,
                "shard_id": args.shard_id,
                "num_shards": args.num_shards,
                "candidate_start": start,
                "candidate_end": stop,
                "results": probe_results,
            }
            write_json_atomic(shard_output, shard_payload)
            print(shard_output)
            return 0

    evidence = build_feasibility_evidence(
        task=args.task,
        candidate_seeds=candidates,
        probe_results=probe_results,
        task_config=args.task_config,
        candidate_partition=args.candidate_partition,
        required_count=args.required_count,
    )
    if args.verify_label:
        evidence["verify_label"] = args.verify_label
    output = args.output or (BRACE_DIR / "runs" / f"seed_feasibility_{args.task}.json")
    write_json_atomic(output, evidence)
    print(json.dumps({
        "task": args.task,
        "task_status": evidence["task_status"],
        "passed": evidence["passed"],
        "solvable_count_in_candidates": evidence["solvable_count_in_candidates"],
        "expert_exception_count": evidence["expert_exception_count"],
        "output": str(output),
    }, indent=2))

    if args.provisional_manifest_update and evidence["task_status"] == TASK_STATUS_PASSED:
        write_provisional_manifest_update(manifest_path, manifest, evidence, output)
        print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
