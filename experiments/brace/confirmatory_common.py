#!/usr/bin/env python3
"""Shared helpers for confirmatory census/cohort/eval provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.brace.replay_audit import read_json


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_set_sha256(seeds: list[int]) -> str:
    payload = json.dumps(sorted(int(seed) for seed in seeds), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def success_counts_by_seed(rows: list[dict[str, Any]], split: str) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in rows:
        if row.get("split") != split:
            continue
        seed = int(row["env_seed"])
        if bool(row.get("success", False)):
            counts[seed] = counts.get(seed, 0) + 1
    return counts


def split_success_rates(eval_path: Path, split: str) -> dict[int, float]:
    rows = read_json(eval_path).get("rows", [])
    by_seed: dict[int, list[bool]] = {}
    for row in rows:
        if row.get("split") != split:
            continue
        by_seed.setdefault(int(row["env_seed"]), []).append(bool(row.get("success", False)))
    return {seed: sum(values) / len(values) for seed, values in by_seed.items()}


def enrollment_eligible(success_count: int, *, min_successes: int) -> bool:
    return success_count >= min_successes


def eligible_seeds_from_eval(
    eval_path: Path,
    *,
    split: str,
    min_successes: int,
    repeats_required: int,
) -> list[int]:
    rows = read_json(eval_path).get("rows", [])
    counts = success_counts_by_seed(rows, split)
    eligible = []
    for seed in sorted(counts):
        total = sum(1 for row in rows if row.get("split") == split and int(row["env_seed"]) == seed)
        if total != repeats_required:
            continue
        if enrollment_eligible(counts[seed], min_successes=min_successes):
            eligible.append(seed)
    return eligible


def repeat_completeness(
    rows: list[dict[str, Any]],
    split: str,
    seeds: list[int],
    repeats_required: int,
    *,
    policy_seed_offset: int | None = None,
) -> bool:
    expected_seeds = {int(seed) for seed in seeds}
    seen: set[tuple[int, int]] = set()
    expected_repeats = set(range(repeats_required))
    for seed in seeds:
        matching = [row for row in rows if row.get("split") == split and int(row["env_seed"]) == int(seed)]
        repeats = [int(row.get("repeat", -1)) for row in matching]
        if len(repeats) != repeats_required or set(repeats) != expected_repeats:
            return False
        for row, repeat in zip(matching, repeats):
            key = (int(seed), repeat)
            if key in seen:
                return False
            seen.add(key)
            if policy_seed_offset is not None and int(row.get("policy_seed", -1)) != policy_seed_offset + repeat:
                return False
    actual_seeds = {
        int(row["env_seed"])
        for row in rows
        if row.get("split") == split
    }
    if actual_seeds != expected_seeds:
        return False
    return True


def feasibility_projection(
    success_counts: dict[int, int],
    *,
    excluded: set[int],
    min_untouched: int,
    repeats_required: int,
) -> dict[str, Any]:
    projections: dict[str, Any] = {}
    for rule_min in (1, 2, 3):
        eligible = [
            seed
            for seed, count in success_counts.items()
            if count >= rule_min and seed not in excluded and count <= repeats_required
        ]
        projections[f"{rule_min}_of_{repeats_required}"] = {
            "eligible_before_cap": len(eligible),
            "untouched_selected": min(len(eligible), min_untouched),
            "meets_min_untouched": len(eligible) >= min_untouched,
        }
    return projections


def validate_census_summary(
    census_summary: dict[str, Any],
    *,
    protocol_path: Path,
    protocol: dict[str, Any],
) -> None:
    expected_offset = int(protocol["census_eval"]["policy_seed_offset"])
    checks = {
        "stage": "confirmatory_base_census",
        "run_type": "confirmatory_base_census",
        "policy_seed_offset": expected_offset,
        "protocol_revision": protocol["protocol_revision"],
    }
    for key, expected in checks.items():
        if census_summary.get(key) != expected:
            raise ValueError(f"census summary {key}={census_summary.get(key)!r}, expected {expected!r}")
    task = census_summary.get("task")
    if task not in protocol.get("tasks", []):
        raise ValueError(f"census summary task {task!r} is not frozen in the protocol")
    if census_summary.get("protocol_sha256") != file_sha256(protocol_path):
        raise ValueError("census summary protocol_sha256 mismatch")
    enrollment = protocol["base_solved_enrollment"]
    expected_rule = {
        "min_successes": int(enrollment["min_successes"]),
        "repeats_required": int(enrollment["repeats_required"]),
        "display_fraction": enrollment.get("display_fraction"),
    }
    if census_summary.get("enrollment_rule") != expected_rule:
        raise ValueError("census summary enrollment_rule mismatch")
    for path_key, sha_key in (
        ("eval_path", "eval_sha256"),
        ("base_checkpoint", "base_checkpoint_sha256"),
        ("seeds_file", "seeds_file_sha256"),
    ):
        path = Path(str(census_summary.get(path_key, "")))
        if not path.is_file() or census_summary.get(sha_key) != file_sha256(path):
            raise ValueError(f"census summary {path_key}/{sha_key} mismatch")
    if not census_summary.get("repeat_completeness", {}).get("id_heldout"):
        raise ValueError("census summary missing complete id_heldout repeats")
    if not census_summary.get("repeat_completeness", {}).get("train_seen"):
        raise ValueError("census summary missing complete train_seen repeats")
    eval_payload = read_json(Path(census_summary["eval_path"]))
    progress = eval_payload.get("progress", {})
    if not progress.get("complete") or int(progress.get("policy_seed_offset", -1)) != expected_offset:
        raise ValueError("census eval progress is incomplete or has wrong policy seed offset")
    id_seeds = [int(seed) for seed in census_summary.get("id_seeds", [])]
    train_seeds = [int(seed) for seed in census_summary.get("train_seeds", [])]
    if not id_seeds or not train_seeds:
        raise ValueError("census summary must freeze exact id_seeds and train_seeds")
    repeats = int(enrollment["repeats_required"])
    rows = eval_payload.get("rows", [])
    if not repeat_completeness(
        rows, "id_heldout", id_seeds, repeats, policy_seed_offset=expected_offset
    ) or not repeat_completeness(
        rows, "train_seen", train_seeds, repeats, policy_seed_offset=expected_offset
    ):
        raise ValueError("census eval rows do not match frozen seeds/repeats/policy seeds")
