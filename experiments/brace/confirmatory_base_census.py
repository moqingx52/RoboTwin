#!/usr/bin/env python3
"""Run confirmatory base census at offset 3000 (ID+train only, no Hard)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
PHASE1_DIR = REPO_ROOT / "experiments" / "phase1"
DP_DIR = REPO_ROOT / "policy" / "DP"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.anchor_behavior_eval import base_checkpoint, build_eval_command
from experiments.brace.confirmatory_common import (
    census_enrollment_split,
    feasibility_projection,
    file_sha256,
    load_census_candidate_ids,
    load_id_heldout_seeds,
    repeat_completeness,
    seed_set_sha256,
    success_counts_by_seed,
)
from experiments.brace.replay_audit import read_json, repo_path, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="place_container_plate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=BRACE_DIR / "screen_protocol.v1.4.2.confirmatory_preservation.json",
    )
    parser.add_argument(
        "--seeds-file",
        type=Path,
        default=BRACE_DIR / "seeds/place_container_plate_confirmatory_v1.4.2_seeds.json",
    )
    parser.add_argument("--hard-seeds-file", type=Path, default=PHASE1_DIR / "eval_results_200/hard_eval_seeds/place_container_plate.json")
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    protocol_path = repo_path(args.protocol)
    protocol = read_json(protocol_path)
    if protocol.get("status") != "frozen" or bool(protocol.get("exploratory", True)):
        raise SystemExit(f"confirmatory census requires frozen protocol: {protocol_path}")
    if args.task not in protocol.get("tasks", []):
        raise SystemExit(f"task {args.task!r} is not frozen in {protocol_path}")

    output_dir = repo_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds_file = repo_path(args.seeds_file)
    hard_seeds_file = repo_path(args.hard_seeds_file)
    for required in (seeds_file, hard_seeds_file):
        if not required.is_file():
            raise SystemExit(f"missing census input: {required}")
    ckpt = base_checkpoint(args.task)
    if not ckpt.is_file():
        raise SystemExit(f"missing base checkpoint: {ckpt}")

    census_cfg = protocol["census_eval"]
    enrollment = protocol["base_solved_enrollment"]
    variant = "census_base"
    command = build_eval_command(
        task=args.task,
        variant=variant,
        checkpoint_path=str(ckpt),
        output_dir=output_dir,
        seeds_file=seeds_file,
        hard_seeds_file=hard_seeds_file,
        extra_splits_file=None,
        workers_per_gpu=args.workers_per_gpu,
        protocol=protocol,
        eval_profile="census",
    )
    env = dict(**__import__("os").environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)

    eval_path = output_dir / args.task / f"{variant}.json"
    rows = read_json(eval_path).get("rows", [])
    seed_payload = read_json(seeds_file)
    census_candidate_ids = load_census_candidate_ids(seed_payload, protocol)
    id_heldout = load_id_heldout_seeds(seed_payload, protocol)
    train_seeds = [int(seed) for seed in seed_payload["train_rollout"]][: int(census_cfg["train_seed_count"])]
    enrollment_split = census_enrollment_split(protocol)
    repeats = int(enrollment["repeats_required"])
    census_counts = success_counts_by_seed(rows, enrollment_split)
    uses_split_schema = "census_candidate_id_count" in census_cfg
    schema_version = 2 if uses_split_schema else 1
    summary: dict[str, Any] = {
        "schema_version": schema_version,
        "stage": "confirmatory_base_census",
        "run_type": "confirmatory_base_census",
        "task": args.task,
        "policy_seed_offset": int(census_cfg["policy_seed_offset"]),
        "protocol_revision": protocol["protocol_revision"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "base_checkpoint": str(ckpt),
        "base_checkpoint_sha256": file_sha256(ckpt),
        "seeds_file": str(seeds_file),
        "seeds_file_sha256": file_sha256(seeds_file),
        "eval_path": str(eval_path),
        "eval_sha256": file_sha256(eval_path),
        "enrollment_rule": {
            "min_successes": int(enrollment["min_successes"]),
            "repeats_required": repeats,
            "display_fraction": enrollment.get("display_fraction"),
        },
        "repeat_completeness": {
            enrollment_split: repeat_completeness(
                rows,
                enrollment_split,
                census_candidate_ids,
                repeats,
                policy_seed_offset=int(census_cfg["policy_seed_offset"]),
            ),
            "train_seen": repeat_completeness(
                rows,
                "train_seen",
                train_seeds,
                repeats,
                policy_seed_offset=int(census_cfg["policy_seed_offset"]),
            ),
        },
        "train_seeds": train_seeds,
        "feasibility_by_rule": feasibility_projection(
            census_counts,
            excluded=set(),
            min_untouched=int(protocol["preservation_cohorts"]["min_untouched_base_solved"]),
            repeats_required=repeats,
        ),
    }
    if uses_split_schema:
        summary["census_candidate_ids"] = census_candidate_ids
        summary["id_heldout"] = id_heldout
        summary["census_seed_set_sha256"] = seed_set_sha256(census_candidate_ids + train_seeds)
        summary["confirmatory_seed_set_sha256"] = seed_set_sha256(id_heldout + train_seeds)
        summary["census_candidate_success_counts"] = {
            str(seed): census_counts.get(seed, 0) for seed in census_candidate_ids
        }
    else:
        summary["id_seeds"] = census_candidate_ids
        summary["seed_set_sha256"] = seed_set_sha256(census_candidate_ids + train_seeds)
        summary["id_success_counts"] = {str(seed): census_counts.get(seed, 0) for seed in census_candidate_ids}
    write_json_atomic(output_dir / "census_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if all(summary["repeat_completeness"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
