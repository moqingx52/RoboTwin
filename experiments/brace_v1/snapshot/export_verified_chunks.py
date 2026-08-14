#!/usr/bin/env python3
"""Export branch-verified (B1) and matched random (N1) chunk datasets from branch artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for import_path in (REPO_ROOT, REPO_ROOT / "experiments" / "phase1"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from experiments.brace.collect_branches import summarize_branch_rows
from experiments.brace.control_trace import load_brace_trace, load_policy_chunk_indices, policy_chunk_actions
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


def exportable_point_types(protocol: dict[str, Any]) -> set[str]:
    configured = (protocol.get("acceptance") or {}).get("accepted_point_types")
    if configured is None:
        configured = ["local_divergence_peak", "first_persistent_divergence"]
    return {str(value) for value in configured}


def accepted_points_from_summary(summary: dict[str, Any], task: str, checks: list[dict[str, Any]], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    # Always recompute: archived summaries may contain the pre-fix candidate-only LCB.
    alpha = float(protocol["acceptance"]["one_sided_alpha"])
    delta = float(protocol["acceptance"]["minimum_advantage_delta"])
    recomputed = summarize_branch_rows(checks, alpha=alpha, delta=delta)
    allowed = exportable_point_types(protocol)
    return [
        point for point in recomputed.get("points", [])
        if point.get("accepted") and str(point.get("point_type")) in allowed
    ]


def branch_confirm_completeness(points: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    gate = protocol.get("branch_confirm_gate") or {}
    min_seeds = int(gate.get("min_seeds", 10))
    min_points = int(gate.get("min_points", 30))
    max_share_allowed = float(gate.get("max_seed_accepted_share", 0.2))
    seeds = {int(point["env_seed"]) for point in points}
    allowed = exportable_point_types(protocol)
    accepted = [
        point for point in points
        if point.get("accepted") and str(point.get("point_type")) in allowed
    ]
    per_seed = Counter(int(point["env_seed"]) for point in accepted)
    max_share = max(per_seed.values(), default=0) / max(1, len(accepted))
    checks = {
        "min_seeds": len(seeds) >= min_seeds,
        "min_points": len(points) >= min_points,
        "max_seed_accepted_share": max_share <= max_share_allowed,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "unique_seeds": len(seeds),
            "total_points": len(points),
            "accepted_points": len(accepted),
            "max_seed_accepted_share": max_share,
        },
        "required": {
            "min_seeds": min_seeds,
            "min_points": min_points,
            "max_seed_accepted_share": max_share_allowed,
        },
    }


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


def _index_success_candidate(candidate: Candidate) -> tuple[str, int, int, str, tuple[int, ...]]:
    indices = tuple(sorted(load_policy_chunk_indices(candidate.path)))
    return (candidate.task, candidate.env_seed, candidate.rollout_id, str(candidate.path), indices)


def index_success_chunk_indices(
    successes: list[Candidate],
    *,
    workers: int,
) -> dict[int, list[Candidate]]:
    if not successes:
        return {}

    chunk_to_candidates: dict[int, list[Candidate]] = defaultdict(list)
    workers = max(1, min(workers, len(successes)))
    print(f"Indexing policy chunk indices for {len(successes)} success trajectories with workers={workers}", flush=True)

    if workers == 1:
        indexed = [_index_success_candidate(candidate) for candidate in successes]
    else:
        indexed = []
        completed = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_index_success_candidate, candidate): candidate for candidate in successes}
            for future in as_completed(futures):
                indexed.append(future.result())
                completed += 1
                if completed == 1 or completed % 25 == 0 or completed == len(successes):
                    print(f"  indexed {completed}/{len(successes)} trajectories", flush=True)

    for task_name, env_seed, rollout_id, path_text, indices in indexed:
        candidate = Candidate(
            task=task_name,
            env_seed=env_seed,
            rollout_id=rollout_id,
            success=True,
            path=Path(path_text),
        )
        for chunk_index in indices:
            chunk_to_candidates[int(chunk_index)].append(candidate)
    return chunk_to_candidates


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
    for point_id, point in enumerate(accepted_points):
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
                "matched_pair_id": point_id,
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
    chunk_to_candidates: dict[int, list[Candidate]],
    selection_seed: int,
    protocol: dict[str, Any],
    run_label: str,
    strict_matching: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not b1_records:
        return [], {"matched": 0, "dropped": 0}

    if strict_matching:
        return build_n1_records_strict(
            task=task,
            b1_records=b1_records,
            chunk_to_candidates=chunk_to_candidates,
            selection_seed=selection_seed,
            protocol=protocol,
            run_label=run_label,
        )

    ckpt = checkpoint_metadata(protocol, task)
    chunk_hist = Counter(int(record["branch_chunk_index"]) for record in b1_records)
    rng = random.Random(selection_seed)
    records: list[dict[str, Any]] = []
    for chunk_index, count in sorted(chunk_hist.items()):
        pool = list(chunk_to_candidates.get(chunk_index, []))
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
    audit = {
        "strict_matching": False,
        "matched": len(records),
        "dropped": 0,
        "b1_unique_env_seeds": sorted({int(row["env_seed"]) for row in b1_records}),
        "n1_unique_env_seeds": sorted({int(row["env_seed"]) for row in records}),
    }
    return records, audit


def build_n1_records_strict(
    *,
    task: str,
    b1_records: list[dict[str, Any]],
    chunk_to_candidates: dict[int, list[Candidate]],
    selection_seed: int,
    protocol: dict[str, Any],
    run_label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ckpt = checkpoint_metadata(protocol, task)
    rng = random.Random(selection_seed)
    records: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    fallback_count = 0
    duplicate_count = 0
    credit_cfg = protocol.get("credit_matching", {})
    max_share = float(credit_cfg.get("max_chunk_share_per_seed", 0.25))
    fail_closed = bool(credit_cfg.get("fail_closed_on_env_seed_miss", True))
    exclude_b1 = bool(credit_cfg.get("exclude_b1_trajectory", True))

    def b1_trajectory_key(record: dict[str, Any]) -> tuple[int, int, int, str]:
        return (
            int(record["env_seed"]),
            int(record["success_rollout_id"]),
            int(record["branch_chunk_index"]),
            str(record["hdf5_path"]),
        )

    used_n1_keys: set[tuple[int, int, int, str]] = set()

    for pair_id, b1 in enumerate(b1_records):
        chunk_index = int(b1["branch_chunk_index"])
        target_seed = int(b1["env_seed"])
        b1_key = b1_trajectory_key(b1)
        pool = [
            candidate
            for candidate in chunk_to_candidates.get(chunk_index, [])
            if candidate.success and int(candidate.env_seed) == target_seed
        ]
        if exclude_b1:
            pool = [
                candidate
                for candidate in pool
                if (int(candidate.env_seed), int(candidate.rollout_id), chunk_index, str(candidate.path)) != b1_key
            ]
        if not pool:
            if fail_closed:
                dropped.append(
                    {
                        "matched_pair_id": pair_id,
                        "b1_env_seed": target_seed,
                        "chunk_index": chunk_index,
                        "reason": "env_seed_pool_empty",
                    }
                )
                continue
            fallback_count += 1
            pool = [
                candidate
                for candidate in chunk_to_candidates.get(chunk_index, [])
                if candidate.success
                and (not exclude_b1 or (int(candidate.env_seed), int(candidate.rollout_id), chunk_index, str(candidate.path)) != b1_key)
            ]
        if not pool:
            dropped.append(
                {
                    "matched_pair_id": pair_id,
                    "b1_env_seed": target_seed,
                    "chunk_index": chunk_index,
                    "reason": "empty_pool",
                }
            )
            continue
        rng.shuffle(pool)
        candidate = pool[0]
        n1_key = (int(candidate.env_seed), int(candidate.rollout_id), chunk_index, str(candidate.path))
        if n1_key in used_n1_keys or n1_key == b1_key:
            duplicate_count += 1
            dropped.append(
                {
                    "matched_pair_id": pair_id,
                    "b1_env_seed": target_seed,
                    "chunk_index": chunk_index,
                    "reason": "duplicate_or_same_as_b1",
                }
            )
            continue
        used_n1_keys.add(n1_key)
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
                "matched_pair_id": pair_id,
                "matched_b1_env_seed": target_seed,
                "exact_env_seed_match": int(candidate.env_seed) == target_seed,
                "interaction_transitions": int(actions.shape[0]),
                **ckpt,
            }
        )

    b1_per_seed = Counter(int(row["env_seed"]) for row in b1_records)
    n1_per_seed = Counter(int(row["env_seed"]) for row in records)
    # Discrete budget-matching: max share is an integer cap ceil(max_share * n),
    # not a raw ratio test (which is infeasible for small n, e.g. 1/3 > 0.25).
    b1_n = max(1, len(b1_records))
    n1_n = max(1, len(records))
    b1_max_allowed = max(1, math.ceil(max_share * b1_n))
    n1_max_allowed = max(1, math.ceil(max_share * n1_n))
    b1_over = {seed: count for seed, count in b1_per_seed.items() if count > b1_max_allowed}
    over_limit = {seed: count for seed, count in n1_per_seed.items() if count > n1_max_allowed}
    exact_pair_matches = sum(1 for row in records if row.get("exact_env_seed_match"))
    audit = {
        "strict_matching": True,
        "matched": len(records),
        "dropped": len(dropped),
        "dropped_rows": dropped,
        "fallback_count": fallback_count,
        "duplicate_count": duplicate_count,
        "exact_pair_matches": exact_pair_matches,
        "b1_unique_env_seeds": sorted(b1_per_seed.keys()),
        "n1_unique_env_seeds": sorted(n1_per_seed.keys()),
        "shared_env_seeds": sorted(set(n1_per_seed) & set(b1_per_seed)),
        "per_seed_chunk_count_b1": dict(b1_per_seed),
        "per_seed_chunk_count_n1": dict(n1_per_seed),
        "max_chunk_share_per_seed": max_share,
        "share_cap_rule": "ceil_max_share_times_n",
        "b1_max_allowed_per_seed": b1_max_allowed,
        "n1_max_allowed_per_seed": n1_max_allowed,
        "over_share_limit_b1": b1_over,
        "over_share_limit_n1": over_limit,
        "b1_chunk_index_hist": dict(Counter(int(row["branch_chunk_index"]) for row in b1_records)),
        "n1_chunk_index_hist": dict(Counter(int(row["branch_chunk_index"]) for row in records)),
        "passed": len(records) == len(b1_records) and not b1_over and not over_limit and duplicate_count == 0,
    }
    if not audit["passed"]:
        raise ValueError(f"strict N1 matching audit failed: {json.dumps(audit, sort_keys=True)}")
    return records, audit


def export_datasets(
    *,
    task: str,
    branch_dir: Path,
    rollout_dir: Path,
    protocol: dict[str, Any],
    run_label: str,
    n1_seed: int,
    workers: int,
) -> dict[str, Any]:
    summary = read_json(branch_dir / "summary.json")
    checks = read_jsonl(branch_dir / "checks.jsonl")
    alpha = float(protocol["acceptance"]["one_sided_alpha"])
    delta = float(protocol["acceptance"]["minimum_advantage_delta"])
    recomputed = summarize_branch_rows(checks, alpha=alpha, delta=delta)
    completeness = branch_confirm_completeness(recomputed.get("points", []), protocol)
    if not completeness["passed"]:
        raise ValueError(
            "branch confirm completeness NO-GO; refuse chunk export: "
            + json.dumps(completeness, sort_keys=True)
        )

    accepted_points = accepted_points_from_summary(summary, task, checks, protocol)
    if not accepted_points:
        raise ValueError("branch confirm has no exportable accepted scientific points")
    candidate_rows = _candidate_rows_by_point(checks)
    successes, _ = collect_candidates(task, rollout_dir)
    success_by_key = {(item.env_seed, item.rollout_id): item for item in successes if item.success}
    success_candidates = [item for item in successes if item.success]

    print(f"Building B1 from {len(accepted_points)} accepted points", flush=True)
    b1_records = build_b1_records(
        task=task,
        accepted_points=accepted_points,
        candidate_rows=candidate_rows,
        success_by_key=success_by_key,
        protocol=protocol,
        run_label=run_label,
    )
    chunk_to_candidates = index_success_chunk_indices(success_candidates, workers=workers)
    print(f"Building N1 matched to {len(b1_records)} B1 chunks", flush=True)
    strict_matching = bool(protocol.get("credit_matching", {}).get("match_env_seed", False))
    n1_records, matching_audit = build_n1_records(
        task=task,
        b1_records=b1_records,
        chunk_to_candidates=chunk_to_candidates,
        selection_seed=n1_seed,
        protocol=protocol,
        run_label=run_label,
        strict_matching=strict_matching,
    )

    return {
        "task": task,
        "run_label": run_label,
        "protocol_revision": protocol.get("protocol_revision"),
        "git_commit": git_commit(),
        "accepted_points": len(accepted_points),
        "b1_chunks": len(b1_records),
        "n1_chunks": len(n1_records),
        "indexed_success_trajectories": len(success_candidates),
        "export_workers": workers,
        "recomputed_accepted_points": recomputed["accepted_points"],
        "exportable_point_types": sorted(exportable_point_types(protocol)),
        "branch_confirm_completeness": completeness,
        "matching_audit": matching_audit,
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
    parser.add_argument(
        "--workers",
        type=int,
        default=min(os.cpu_count() or 1, 96),
        help="Parallel workers for N1 chunk-index scanning (HDF5 metadata only).",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    protocol = read_json(repo_path(args.protocol))
    branch_dir = repo_path(args.branch_dir)
    rollout_dir = repo_path(args.rollout_dir)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checks_path = branch_dir / "checks.jsonl"
    if not checks_path.is_file():
        fallback = REPO_ROOT / "experiments" / "brace" / "branches" / "checks.jsonl"
        raise SystemExit(
            f"missing {checks_path}\n"
            f"Archive place pilot without checks.jsonl cannot export B1.\n"
            f"Fix on cloud:\n"
            f"  cp experiments/brace/branches/checks.jsonl {branch_dir}/\n"
            f"  bash experiments/brace/archive_place_pilot.sh\n"
            f"Or export from working copy:\n"
            f"  BRACE_BRANCH_DIR=experiments/brace/branches ... export-verified-chunks"
            + (f"\n(found fallback at {fallback})" if fallback.is_file() else "")
        )

    payload = export_datasets(
        task=args.task,
        branch_dir=branch_dir,
        rollout_dir=rollout_dir,
        protocol=protocol,
        run_label=args.run_label,
        n1_seed=args.n1_seed,
        workers=args.workers,
    )
    if payload.get("matching_audit") and not payload["matching_audit"].get("passed", True):
        raise SystemExit(f"matching audit failed; refusing to promote manifests: {payload['matching_audit']}")
    write_jsonl_atomic(output_dir / f"{args.run_label}_B1.jsonl", payload["records"]["B1"])
    write_jsonl_atomic(output_dir / f"{args.run_label}_N1.jsonl", payload["records"]["N1"])
    if payload.get("matching_audit"):
        write_json_atomic(output_dir / f"{args.run_label}_matching_audit.json", payload["matching_audit"])
    summary = {key: value for key, value in payload.items() if key != "records"}
    write_json_atomic(output_dir / f"{args.run_label}_summary.json", summary)
    try:
        from experiments.brace.stage_records import emit_stage_record

        emit_stage_record(
            "export_verified_chunks",
            summary=summary,
            summary_path=output_dir / f"{args.run_label}_summary.json",
            tasks=[args.task],
            label=args.run_label,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to emit stage record: {exc}", file=sys.stderr)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
