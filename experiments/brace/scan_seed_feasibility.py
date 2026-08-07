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

from experiments.brace.multitask_protocol import SUPPLEMENT_SELECTION_RULE, file_sha256
from experiments.brace.replay_audit import read_json, write_json_atomic
from experiments.brace.seed_feasibility import (
    DEFAULT_CANDIDATE_PARTITION,
    DEFAULT_EXPERT_DEMO_COUNT,
    DEFAULT_TASK_CONFIG,
    PROBE_SUCCESS_RULES,
    PROBE_SUCCESS_RULE_SINGLE,
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


def validate_shard_payload(payload: dict, path: Path, *, candidate_count: int, task: str) -> None:
    """A shard file must self-describe its exact candidate range."""
    missing = [key for key in ("task", "shard_id", "num_shards", "candidate_start", "candidate_end", "results") if key not in payload]
    if missing:
        raise ValueError(f"{path}: shard payload missing {missing}")
    if payload.get("task") != task:
        raise ValueError(f"{path}: shard task {payload.get('task')} != {task}")
    shard_id = int(payload["shard_id"])
    num_shards = int(payload["num_shards"])
    start, stop = shard_slice(candidate_count, shard_id, num_shards)
    if (int(payload["candidate_start"]), int(payload["candidate_end"])) != (start, stop):
        raise ValueError(
            f"{path}: shard {shard_id}/{num_shards} claims range "
            f"[{payload['candidate_start']},{payload['candidate_end']}) but expected [{start},{stop})"
        )
    results = payload.get("results", [])
    if len(results) != stop - start:
        raise ValueError(
            f"{path}: shard {shard_id}/{num_shards} has {len(results)} rows, "
            f"expected {stop - start} for range [{start},{stop})"
        )
    for offset, row in enumerate(results):
        expected_seed = int(row.get("seed", -1))
        index = start + offset
        if expected_seed < 0:
            raise ValueError(f"{path}: result row {index} missing seed")
        if "episode_idx" not in row or "passed" not in row:
            raise ValueError(f"{path}: result row for seed {expected_seed} missing probe fields")


def merge_probe_results(paths: list[Path], *, candidate_seeds: list[int], task: str) -> list[dict]:
    """Merge shard results with fail-closed completeness checks.

    The set of shard files must exactly partition the candidate range: every
    (shard_id, num_shards) present exactly once, ranges disjoint and covering
    all candidates, and every candidate seed present exactly once in manifest
    order. Duplicate or stale shard files abort the merge.
    """
    if not paths:
        raise ValueError("no shard files to merge")
    expected: dict[int, list[int]] = {}
    payloads: dict[int, dict] = {}
    num_shards_by_id: dict[int, int] = {}
    for path in paths:
        payload = read_json(path)
        validate_shard_payload(payload, path, candidate_count=len(candidate_seeds), task=task)
        shard_id = int(payload["shard_id"])
        if shard_id in num_shards_by_id and num_shards_by_id[shard_id] != int(payload["num_shards"]):
            raise ValueError(f"{path}: inconsistent num_shards for shard_id {shard_id}")
        if shard_id in payloads:
            raise ValueError(f"{path}: duplicate shard_id {shard_id} (stale shard file?)")
        num_shards_by_id[shard_id] = int(payload["num_shards"])
        payloads[shard_id] = payload
        expected[shard_id] = [int(row["seed"]) for row in payload["results"]]

    num_shards_values = set(num_shards_by_id.values())
    if len(num_shards_values) != 1:
        raise ValueError(f"shard files disagree on num_shards: {sorted(num_shards_values)}")
    num_shards = num_shards_values.pop()
    present = sorted(num_shards_by_id)
    if present != list(range(num_shards)):
        raise ValueError(
            f"shard coverage incomplete: have shard ids {present}, expected 0..{num_shards - 1}"
        )

    merged: list[dict] = []
    seen_seeds: set[int] = set()
    for shard_id in range(num_shards):
        start, stop = shard_slice(len(candidate_seeds), shard_id, num_shards)
        payload = payloads[shard_id]
        for offset, row in enumerate(payload["results"]):
            seed = int(row["seed"])
            index = start + offset
            if seed in seen_seeds:
                raise ValueError(f"duplicate seed {seed} across shards")
            if seed != candidate_seeds[index]:
                raise ValueError(
                    f"seed order mismatch: shard {shard_id} row {offset} has seed {seed}, "
                    f"expected candidate {candidate_seeds[index]}"
                )
            seen_seeds.add(seed)
            merged.append(row)
    if len(seen_seeds) != len(candidate_seeds):
        missing = [seed for seed in candidate_seeds if seed not in seen_seeds]
        raise ValueError(f"merged results do not cover all candidates; missing {missing}")
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
    updated["feasibility"]["passed"] = True
    write_json_atomic(manifest_path, updated)


def load_supplement_original(original_evidence_path: Path, task: str, base_candidates: list[int]) -> dict:
    """Load and verify the original pool evidence a supplement scan extends."""
    original = read_json(original_evidence_path)
    if original.get("task") != task:
        raise SystemExit(
            f"--original-evidence task {original.get('task')} != {task}"
        )
    if original.get("schema_version") != 2 or not original.get("task_status"):
        raise SystemExit(
            f"--original-evidence must be schema v2 with task_status: {original_evidence_path}"
        )
    if original.get("supplement", {}).get("used"):
        raise SystemExit("refusing to supplement an already supplemented evidence")
    original_results = original.get("results", [])
    original_seeds = [int(row["seed"]) for row in original_results]
    if original_seeds != [int(seed) for seed in base_candidates]:
        raise SystemExit(
            f"--original-evidence results do not cover the {task} {original.get('candidate_partition')} "
            "pool exactly in manifest order"
        )
    return original


def build_supplement_block(
    original: dict,
    original_path: Path,
    supplement_candidates: list[int],
    supplement_results: list[dict],
    combined_selected: list[int],
    *,
    supplement_shard_count: int,
) -> dict:
    base_seeds = {int(row["seed"]) for row in original["results"]}
    used = [int(seed) for seed in combined_selected if int(seed) not in base_seeds]
    supplement_solvable = sum(1 for row in supplement_results if row.get("passed") is True)
    return {
        "partition": "expert_demo_supplement",
        "count": len(supplement_candidates),
        "original_pool_solvable": sum(1 for row in original["results"] if row.get("passed") is True),
        "original_pool_candidate_count": len(original["results"]),
        "original_pool_task_status": original.get("task_status"),
        "original_evidence_path": str(
            original_path.resolve().relative_to(REPO_ROOT.resolve())
            if original_path.resolve().is_relative_to(REPO_ROOT.resolve()) else str(original_path)
        ),
        "original_evidence_sha256": file_sha256(original_path),
        "selection_rule": SUPPLEMENT_SELECTION_RULE,
        "supplement_solvable": supplement_solvable,
        "supplement_shard_count": supplement_shard_count,
        "used": bool(used),
        "supplement_seeds_used": used,
    }


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
    parser.add_argument("--gpu-id", default="")
    parser.add_argument("--probe-repeats", type=int, default=1)
    parser.add_argument("--probe-success-rule", default=PROBE_SUCCESS_RULE_SINGLE)
    parser.add_argument(
        "--original-evidence",
        type=Path,
        default=None,
        help="original pool evidence (schema v2); combine it with the scanned "
        "candidate-partition per the brace.multitask.v1.1 supplement rule",
    )
    args = parser.parse_args()

    if args.probe_repeats < 1:
        raise SystemExit("--probe-repeats must be positive")
    if args.probe_success_rule not in PROBE_SUCCESS_RULES:
        raise SystemExit(f"--probe-success-rule must be one of {sorted(PROBE_SUCCESS_RULES)}")

    manifest_path = args.manifest_dir / f"{args.task}.json"
    manifest = read_json(manifest_path)
    candidates = manifest.get("partitions", {}).get(args.candidate_partition, [])
    if not candidates:
        raise SystemExit(f"missing candidate partition {args.candidate_partition} in {manifest_path}")

    original_evidence = None
    if args.original_evidence is not None:
        if not args.original_evidence.is_file():
            raise SystemExit(f"--original-evidence file missing: {args.original_evidence}")
        original_evidence = load_supplement_original(args.original_evidence, args.task, candidates)

    # In combine mode the scanned partition is the supplement pool; the base
    # candidates are only used to verify the bound original evidence.
    scan_candidates = candidates
    if original_evidence is not None:
        supplement_candidates = manifest.get("partitions", {}).get("expert_demo_supplement", [])
        if not supplement_candidates:
            raise SystemExit("missing expert_demo_supplement partition in manifest")
        scan_candidates = [int(seed) for seed in supplement_candidates]

    if args.merge_inputs:
        probe_results = merge_probe_results(
            [Path(path) for path in args.merge_inputs],
            candidate_seeds=scan_candidates,
            task=args.task,
        )
    else:
        start, stop = shard_slice(len(scan_candidates), args.shard_id, args.num_shards)
        probe_results = scan_candidate_seeds(
            args.task,
            scan_candidates,
            task_config=args.task_config,
            start_index=start,
            limit=stop - start,
            probe_repeats=args.probe_repeats,
            probe_success_rule=args.probe_success_rule,
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
                "gpu": args.gpu_id or None,
                "probe_repeats": args.probe_repeats,
                "probe_success_rule": args.probe_success_rule,
                "results": probe_results,
            }
            write_json_atomic(shard_output, shard_payload)
            print(shard_output)
            return 0

    shard_gpus: list[str] = []
    if args.merge_inputs:
        for path in args.merge_inputs:
            shard_payload = read_json(Path(path))
            gpu = shard_payload.get("gpu")
            if gpu and gpu not in shard_gpus:
                shard_gpus.append(str(gpu))
    elif args.gpu_id:
        shard_gpus = [args.gpu_id]
    shard_count = len(args.merge_inputs) if args.merge_inputs else (1 if args.num_shards <= 1 else args.num_shards)

    evidence_kwargs = dict(
        task=args.task,
        task_config=args.task_config,
        candidate_partition=args.candidate_partition,
        required_count=args.required_count,
        manifest_path=manifest_path,
        probe_repeats=args.probe_repeats,
        probe_success_rule=args.probe_success_rule,
    )
    supplement_block = None
    if original_evidence is not None:
        supplement_candidates = manifest.get("partitions", {}).get("expert_demo_supplement", [])
        if len(probe_results) != len(supplement_candidates):
            raise SystemExit(
                f"supplement scan results ({len(probe_results)}) do not cover the "
                f"supplement partition ({len(supplement_candidates)})"
            )
        combined_candidates = list(candidates) + [int(seed) for seed in supplement_candidates]
        combined_results = list(original_evidence["results"]) + list(probe_results)
        evidence_kwargs.update(
            candidate_seeds=combined_candidates,
            probe_results=combined_results,
            gpus=list(dict.fromkeys((original_evidence.get("gpus") or []) + shard_gpus)) or None,
            shard_count=(original_evidence.get("shard_count") or 0) + shard_count,
            source_evidence={
                "path": original_evidence.get("provenance", {}).get("source_evidence", {}).get("path")
                or str(args.original_evidence.resolve()),
                "sha256": file_sha256(args.original_evidence),
            },
        )
    else:
        evidence_kwargs.update(
            candidate_seeds=candidates,
            probe_results=probe_results,
            gpus=shard_gpus or None,
            shard_count=shard_count,
        )

    evidence = build_feasibility_evidence(**evidence_kwargs)
    if original_evidence is not None:
        supplement_block = build_supplement_block(
            original_evidence,
            args.original_evidence,
            manifest.get("partitions", {}).get("expert_demo_supplement", []),
            probe_results,
            evidence["expert_demo_seeds"],
            supplement_shard_count=shard_count,
        )
        evidence["supplement"] = supplement_block
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
        "shard_count": evidence.get("shard_count"),
        "supplement_used": bool(supplement_block and supplement_block["used"]),
        "output": str(output),
    }, indent=2))

    if args.provisional_manifest_update and evidence["task_status"] == TASK_STATUS_PASSED:
        write_provisional_manifest_update(manifest_path, manifest, evidence, output)
        print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
