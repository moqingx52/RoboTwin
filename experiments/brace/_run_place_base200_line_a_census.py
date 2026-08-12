#!/usr/bin/env python3
"""Base200 Line A census and preservation cohort freeze for place_container_plate."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
BRACE = REPO / "experiments" / "brace"
PHASE1 = REPO / "experiments" / "phase1"
DP = REPO / "policy" / "DP"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.brace.confirmatory_common import file_sha256
from experiments.brace.replay_audit import read_json, write_json_atomic
from experiments.brace.select_preservation_cohort import select_preservation_cohort


def build_line_a_seeds(*, manifest_path: Path, output_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    parts = manifest["partitions"]
    payload = {
        "schema_version": 2,
        "task": "place_container_plate",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": file_sha256(manifest_path),
        "census_candidate_id": [int(x) for x in parts["census_candidate"]],
        "eval_id": [int(x) for x in parts["confirm_easy"]],
        "train_rollout": [int(x) for x in parts["rollout_train"]],
        "confirm_hard": [int(x) for x in parts["confirm_hard"]],
        "rollout_train": [int(x) for x in parts["rollout_train"]],
        "status": "frozen",
        "frozen_for": "base200_line_a_preservation",
    }
    write_json_atomic(output_path, payload)
    sidecar = Path(str(output_path) + ".sha256")
    sidecar.write_text(f"{file_sha256(output_path)}  {output_path.name}\n", encoding="utf-8")
    return payload


def collect_line_a_exclusions(*, run_dir: Path, branch_summary: Path, b1_jsonl: Path) -> dict[str, list[int]]:
    anchor_manifest = read_json(
        run_dir / "datasets/place_container_plate_anchor_replay.zarr/brace_anchor_manifest.json"
    )
    anchor_seeds = sorted({int(row["env_seed"]) for row in anchor_manifest.get("episodes", [])})
    base_solved = sorted(
        {
            int(row["env_seed"])
            for row in anchor_manifest.get("episodes", [])
            if str(row.get("preservation_group")) == "base_solved"
        }
    )
    sft = sorted(
        {
            int(json.loads(line)["env_seed"])
            for line in b1_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    )
    branch = read_json(branch_summary)
    branch_seeds = sorted(
        {
            int(row["env_seed"])
            for row in branch["tasks"]["place_container_plate"].get("points", [])
            if row.get("accepted")
        }
    )
    manifest = read_json(BRACE / "seeds/multitask_v1/place_container_plate.json")
    rollout_train = sorted(int(x) for x in manifest["partitions"]["rollout_train"])
    return {
        "anchor_train": anchor_seeds,
        "anchor_probe": [],
        "sft_chunk": sft,
        "branch_pilot": branch_seeds,
        "branch_confirm": [],
        "phase3c_behavior": base_solved,
        "rollout_train": rollout_train,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=BRACE / "screen_protocol.v1.2.base200_line_a_preservation.json",
    )
    parser.add_argument("--skip-census", action="store_true")
    parser.add_argument("--skip-cohort", action="store_true")
    args = parser.parse_args()

    run_ptr = BRACE / "runs/LATEST_place_base200_v2_line_a_pilot"
    run_dir = args.run_dir or Path(run_ptr.read_text(encoding="utf-8").strip())
    if not run_dir.is_absolute():
        run_dir = REPO / run_dir
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = args.protocol.resolve()
    protocol = read_json(protocol_path)
    seeds_path = BRACE / "seeds/place_container_plate_base200_line_a_seeds.json"
    build_line_a_seeds(
        manifest_path=BRACE / "seeds/multitask_v1/place_container_plate.json",
        output_path=seeds_path,
    )
    hard_seeds = PHASE1 / "eval_results_200/hard_eval_seeds/place_container_plate.json"
    census_eval = output_dir / "place_container_plate/census_base.json"
    census_summary_path = output_dir / "census_summary.json"

    if not args.skip_census:
        cmd = [
            sys.executable,
            str(BRACE / "confirmatory_base_census.py"),
            "--task",
            "place_container_plate",
            "--output",
            str(output_dir),
            "--protocol",
            str(protocol_path),
            "--seeds-file",
            str(seeds_path),
            "--hard-seeds-file",
            str(hard_seeds),
            "--gpu",
            str(args.gpu),
            "--workers-per-gpu",
            str(args.workers_per_gpu),
        ]
        subprocess.run(cmd, cwd=REPO, check=True)
    if not census_summary_path.is_file() or not census_eval.is_file():
        raise SystemExit(f"missing census artifacts under {output_dir}")

    cohort_path = output_dir / "place_container_plate_preservation_cohort.json"
    if not args.skip_cohort:
        census_summary = read_json(census_summary_path)
        exclusions = collect_line_a_exclusions(
            run_dir=run_dir,
            branch_summary=BRACE / "archive/branches_place_base200_v2/summary.json",
            b1_jsonl=BRACE / "datasets/place_base200_v2_B1.jsonl",
        )
        payload = select_preservation_cohort(
            task="place_container_plate",
            base_eval=census_eval,
            census_summary=census_summary,
            seeds_file=seeds_path,
            exclusions=exclusions,
            enrollment_rule=protocol["base_solved_enrollment"],
            min_untouched=int(protocol["preservation_cohorts"]["min_untouched_base_solved"]),
        )
        payload["run_type"] = "base200_line_a_preservation_cohort"
        payload["protocol_path"] = str(protocol_path)
        payload["protocol_sha256"] = file_sha256(protocol_path)
        payload["protocol_revision"] = protocol["protocol_revision"]
        payload["input_sha256"] = {
            "seeds_file": file_sha256(seeds_path),
            "census_summary": file_sha256(census_summary_path),
            "census_eval": file_sha256(census_eval),
        }
        payload["exclusions"]["rollout_train"] = exclusions["rollout_train"]
        excluded = set()
        for values in exclusions.values():
            excluded.update(int(seed) for seed in values)
        payload["excluded_union_count"] = len(excluded)
        write_json_atomic(cohort_path, payload)
        pointer = BRACE / "runs/LATEST_place_base200_line_a_preservation_cohort"
        pointer.write_text(str(cohort_path) + "\n", encoding="utf-8")
        if not payload.get("meets_min_untouched"):
            print(json.dumps({"warning": "cohort below min_untouched", "payload": payload}, indent=2))
            return 2
    print(json.dumps({"census_summary": str(census_summary_path), "cohort": str(cohort_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
