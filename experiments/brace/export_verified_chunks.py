#!/usr/bin/env python3
"""Export branch-verified (B1) and matched random (N1) chunk datasets from branch artifacts."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for import_path in (REPO_ROOT, REPO_ROOT / "experiments" / "phase1"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from experiments.brace.collect_branches import summarize_branch_rows
from experiments.brace.control_trace import load_brace_trace, policy_chunk_actions
from experiments.brace.replay_audit import (
    Candidate,
    collect_candidates,
    file_sha256,
    git_commit,
    read_json,
    repo_path,
    write_json_atomic,
    write_jsonl_atomic,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def accepted_points_from_summary(summary: dict[str, Any], task: str, checks: list[dict[str, Any]], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    task_summary = summary.get("tasks", {}).get(task, {})
    points = task_summary.get("points")
    if points:
        return [point for point in points if point.get("accepted")]
    alpha = float(protocol["acceptance"]["one_sided_alpha"])
    delta = float(protocol["acceptance"]["minimum_advantage_delta"])
    recomputed = summarize_branch_rows(checks, alpha=alpha, delta=delta)
    return [point for point in recomputed.get("points", []) if point.get("accepted")]


def checkpoint_metadata(protocol: dict[str, Any], task: str) -> dict[str, Any]:
    pattern = protocol.get("base_checkpoint_pattern", "")
    checkpoint_path = pattern.format(task=task)
    path = repo_path(checkpoint_path)
    return {
        "checkpoint_path": str(path),
        "checkpoint_sha256": file_sha256(path) if path.is_file() else None,
    }


def _candidate_rows_by_point(rows: list[dict[str, Any]]) -> dict[tuple[int, int, int], dict[str, Any]]:
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("branch_role") != "candidate":
            continue
        key = (int(row["env_seed"]), int(row["snapshot_id"]), int(row["physics_step"]))
        grouped[key].append(row)
    result: dict[tuple[int, int, int], dict[str, Any]] = {}
    for key, group in grouped.items():
        preferred = next((row for row in group if int(row.get("continuation_seed", -1)) == 0), group[0])
        result[key] = preferred
    return result


def build_b1_records(
    *,
    task: str,
    accepted_points: list[dict[str, Any]],
    candidate_rows: dict[tuple[int, int, int], dict[str, Any]],
    success_by_key: dict[tuple[int, int], Candidate],
    protocol: dict[str, Any],
    run_label: str,
) -> list[dict[str, Any]]:
    ckpt = checkpoint_metadata(protocol, task)
    records: list[dict[str, Any]] = []
    for point in accepted_points:
        key = (int(point["env_seed"]), int(point["snapshot_id"]), int(point["physics_step"]))
        row = candidate_rows.get(key)
        if row is None:
            continue
        success = success_by_key[(int(row["env_seed"]), int(row["success_rollout_id"]))]
        trace = load_brace_trace(success.path)
        chunk_index = int(point["branch_chunk_index"])
        chunk = next(item for item in trace["policy_chunks"] if int(item["chunk_index"]) == chunk_index)
        actions = policy_chunk_actions(chunk["action"])
        records.append(
            {
                "dataset": "B1",
                "run_label": run_label,
                "task": task,
                "env_seed": int(point["env_seed"]),
                "snapshot_id": int(point["snapshot_id"]),
                "physics_step": int(point["physics_step"]),
                "snapshot_physics_step": int(point["snapshot_physics_step"]),
                "branch_chunk_index": chunk_index,
                "success_rollout_id": int(row["success_rollout_id"]),
                "continuation_seed": int(row["continuation_seed"]),
                "policy_seed": int(row["success_rollout_id"]),
                "hdf5_path": str(success.path),
                "chunk_action_shape": list(actions.shape),
                "advantage": float(point["advantage"]),
                "lcb_advantage": float(point["lcb_advantage"]),
                "candidate_success_rate": float(point["candidate_success_rate"]),
                "control_success_rate": float(point["control_success_rate"]),
                "interaction_transitions": int(row.get("transitions", 0)),
                **ckpt,
            }
        )
    return records


def build_n1_records(
    *,
    task: str,
    b1_records: list[dict[str, Any]],
    rollout_dir: Path,
    selection_seed: int,
    protocol: dict[str, Any],
    run_label: str,
) -> list[dict[str, Any]]:
    if not b1_records:
        return []

    ckpt = checkpoint_metadata(protocol, task)
    candidates, _ = collect_candidates(task, rollout_dir)
    successes = [item for item in candidates if item.success]
    if not successes:
        raise ValueError(f"no success trajectories for N1 sampling in {task}")

    chunk_hist = Counter(int(record["branch_chunk_index"]) for record in b1_records)
    rng = random.Random(selection_seed)
    records: list[dict[str, Any]] = []
    for chunk_index, count in sorted(chunk_hist.items()):
        pool = []
        for candidate in successes:
            trace = load_brace_trace(candidate.path)
            if any(int(chunk["chunk_index"]) == chunk_index for chunk in trace["policy_chunks"]):
                pool.append(candidate)
        if not pool:
            raise ValueError(f"no success trajectories with chunk_index={chunk_index} for N1")
        rng.shuffle(pool)
        for pick_index in range(count):
            candidate = pool[pick_index % len(pool)]
            trace = load_brace_trace(candidate.path)
            chunk = next(item for item in trace["policy_chunks"] if int(item["chunk_index"]) == chunk_index)
            actions = policy_chunk_actions(chunk["action"])
            records.append(
                {
                    "dataset": "N1",
                    "run_label": run_label,
                    "task": task,
                    "env_seed": int(candidate.env_seed),
                    "branch_chunk_index": int(chunk_index),
                    "success_rollout_id": int(candidate.rollout_id),
                    "policy_seed": int(candidate.rollout_id),
                    "hdf5_path": str(candidate.path),
                    "chunk_action_shape": list(actions.shape),
                    "matched_b1_chunk_index": int(chunk_index),
                    "interaction_transitions": int(actions.shape[0]),
                    **ckpt,
                }
            )
    return records


def export_datasets(
    *,
    task: str,
    branch_dir: Path,
    rollout_dir: Path,
    protocol: dict[str, Any],
    run_label: str,
    n1_seed: int,
) -> dict[str, Any]:
    summary = read_json(branch_dir / "summary.json")
    checks = read_jsonl(branch_dir / "checks.jsonl")
    alpha = float(protocol["acceptance"]["one_sided_alpha"])
    delta = float(protocol["acceptance"]["minimum_advantage_delta"])
    recomputed = summarize_branch_rows(checks, alpha=alpha, delta=delta)

    accepted_points = accepted_points_from_summary(summary, task, checks, protocol)
    candidate_rows = _candidate_rows_by_point(checks)
    successes, _ = collect_candidates(task, rollout_dir)
    success_by_key = {(item.env_seed, item.rollout_id): item for item in successes if item.success}

    b1_records = build_b1_records(
        task=task,
        accepted_points=accepted_points,
        candidate_rows=candidate_rows,
        success_by_key=success_by_key,
        protocol=protocol,
        run_label=run_label,
    )
    n1_records = build_n1_records(
        task=task,
        b1_records=b1_records,
        rollout_dir=rollout_dir,
        selection_seed=n1_seed,
        protocol=protocol,
        run_label=run_label,
    )

    return {
        "task": task,
        "run_label": run_label,
        "protocol_revision": protocol.get("protocol_revision"),
        "git_commit": git_commit(),
        "accepted_points": len(accepted_points),
        "b1_chunks": len(b1_records),
        "n1_chunks": len(n1_records),
        "recomputed_accepted_points": recomputed["accepted_points"],
        "records": {"B1": b1_records, "N1": n1_records},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("experiments/brace/protocol.v2.3.json"))
    parser.add_argument("--branch-dir", type=Path, default=Path("experiments/brace/branches"))
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n1-seed", type=int, default=0)
    args = parser.parse_args()

    protocol = read_json(repo_path(args.protocol))
    branch_dir = repo_path(args.branch_dir)
    rollout_dir = repo_path(args.rollout_dir)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = export_datasets(
        task=args.task,
        branch_dir=branch_dir,
        rollout_dir=rollout_dir,
        protocol=protocol,
        run_label=args.run_label,
        n1_seed=args.n1_seed,
    )
    write_jsonl_atomic(output_dir / f"{args.run_label}_B1.jsonl", payload["records"]["B1"])
    write_jsonl_atomic(output_dir / f"{args.run_label}_N1.jsonl", payload["records"]["N1"])
    summary = {key: value for key, value in payload.items() if key != "records"}
    write_json_atomic(output_dir / f"{args.run_label}_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
