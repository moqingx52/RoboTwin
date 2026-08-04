#!/usr/bin/env python3
"""Select disjoint preservation/boundary cohorts for confirmatory eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
PHASE1_DIR = REPO_ROOT / "experiments" / "phase1"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.anchor_behavior_eval import find_anchor_manifest
from experiments.brace.confirmatory_common import (
    eligible_seeds_from_eval,
    feasibility_projection,
    file_sha256,
    seed_set_sha256,
    split_success_rates,
    success_counts_by_seed,
    validate_census_summary,
)
from experiments.brace.replay_audit import read_json, repo_path, write_json_atomic


def load_seed_list(path: Path) -> list[int]:
    payload = read_json(path)
    if "seeds" in payload:
        return [int(seed) for seed in payload["seeds"]]
    if "analyzed_seeds" in payload:
        return [int(seed) for seed in payload["analyzed_seeds"]]
    raise ValueError(f"unsupported seed list format: {path}")


def load_sft_chunk_seeds(dataset_manifest: Path | None) -> set[int]:
    if dataset_manifest is None or not dataset_manifest.is_file():
        return set()
    if dataset_manifest.suffix == ".jsonl":
        rows = [json.loads(line) for line in dataset_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        return {int(row["env_seed"]) for row in rows if "env_seed" in row}
    payload = read_json(dataset_manifest)
    seeds: set[int] = set()
    for row in payload.get("chunks", payload.get("episodes", [])):
        if "env_seed" in row:
            seeds.add(int(row["env_seed"]))
    return seeds


def load_census_artifacts(census_dir: Path) -> tuple[dict[str, Any], Path]:
    summary_path = census_dir / "census_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing census summary: {summary_path}")
    summary = read_json(summary_path)
    eval_path = Path(summary["eval_path"])
    if not eval_path.is_file():
        raise FileNotFoundError(f"missing census eval artifact: {eval_path}")
    return summary, eval_path


def collect_exclusions(
    *,
    task: str,
    calibration_run_dir: Path,
    behavior_eval_dir: Path | None,
    pilot_seeds_file: Path | None,
    confirm_seeds_file: Path | None,
    dataset_manifest: Path | None,
) -> dict[str, list[int]]:
    probe_split_path = calibration_run_dir / "A1" / "anchor_probe_split.json"
    probe_split = read_json(probe_split_path) if probe_split_path.is_file() else {}
    manifest_path = find_anchor_manifest(calibration_run_dir, task=task)
    manifest = read_json(manifest_path)
    anchor_train = sorted(int(seed) for seed in probe_split.get("train_env_seeds", []))
    anchor_probe = sorted(int(seed) for seed in probe_split.get("probe_env_seeds", []))
    phase3c_behavior = sorted(
        {
            int(row["env_seed"])
            for row in manifest.get("episodes", [])
            if str(row.get("preservation_group")) == "base_solved"
        }
    )
    if behavior_eval_dir is not None:
        split_path = behavior_eval_dir / "base_solved_anchor_split.json"
        if split_path.is_file():
            phase3c_behavior = sorted(
                set(phase3c_behavior) | {int(seed) for seed in read_json(split_path).get("base_solved_anchor", [])}
            )
        behavior_paths: list[Path] = []
        behavior_summary_path = behavior_eval_dir / "summary.json"
        if behavior_summary_path.is_file():
            for candidate in read_json(behavior_summary_path).get("candidates", []):
                eval_output = candidate.get("eval_output")
                if eval_output:
                    behavior_paths.append(Path(str(eval_output)))
        fallback_base = behavior_eval_dir / task / "calib_base_original.json"
        if fallback_base.is_file():
            behavior_paths.append(fallback_base)
        for eval_path in behavior_paths:
            if eval_path.is_file():
                phase3c_behavior = sorted(
                    set(phase3c_behavior)
                    | {
                        int(row["env_seed"])
                        for row in read_json(eval_path).get("rows", [])
                        if "env_seed" in row
                    }
                )
    return {
        "anchor_train": anchor_train,
        "anchor_probe": anchor_probe,
        "sft_chunk": sorted(load_sft_chunk_seeds(dataset_manifest)),
        "branch_pilot": sorted(load_seed_list(pilot_seeds_file)) if pilot_seeds_file and pilot_seeds_file.is_file() else [],
        "branch_confirm": sorted(load_seed_list(confirm_seeds_file)) if confirm_seeds_file and confirm_seeds_file.is_file() else [],
        "phase3c_behavior": phase3c_behavior,
    }


def select_preservation_cohort(
    *,
    task: str,
    base_eval: Path,
    census_summary: dict[str, Any],
    seeds_file: Path,
    exclusions: dict[str, list[int]],
    enrollment_rule: dict[str, Any],
    min_untouched: int = 60,
    boundary_count: int = 20,
) -> dict[str, Any]:
    seed_payload = read_json(seeds_file)
    all_train = [int(seed) for seed in census_summary.get("train_seeds", seed_payload.get("train_rollout", []))]
    if census_summary.get("schema_version", 1) >= 2:
        census_candidate_ids = [int(seed) for seed in census_summary.get("census_candidate_ids", [])]
        all_id = [int(seed) for seed in census_summary.get("id_heldout", [])]
        enrollment_split = "census_candidate_id"
    else:
        census_candidate_ids = [
            int(seed)
            for seed in census_summary.get(
                "id_seeds", seed_payload.get("eval_id", seed_payload.get("id_heldout", []))
            )
        ]
        all_id = list(census_candidate_ids)
        enrollment_split = "id_heldout"
    excluded = set()
    for values in exclusions.values():
        excluded.update(int(seed) for seed in values)

    min_successes = int(enrollment_rule["min_successes"])
    repeats_required = int(enrollment_rule["repeats_required"])
    rows = read_json(base_eval).get("rows", [])
    id_counts = success_counts_by_seed(rows, enrollment_split)
    base_solved_candidates = eligible_seeds_from_eval(
        base_eval,
        split=enrollment_split,
        min_successes=min_successes,
        repeats_required=repeats_required,
    )
    base_solved_candidates = [seed for seed in base_solved_candidates if seed not in excluded]
    untouched = base_solved_candidates[:min_untouched]
    train_rates = split_success_rates(base_eval, "train_seen")
    boundary_pool = [seed for seed in all_train if seed in train_rates and seed not in excluded and seed not in untouched]
    boundary = sorted(boundary_pool, key=lambda seed: (abs(train_rates[seed] - 0.5), seed))[:boundary_count]

    return {
        "task": task,
        "schema_version": 2,
        "run_type": "confirmatory_preservation_cohort",
        "frozen": True,
        "census_eval_path": census_summary.get("eval_path"),
        "census_policy_seed_offset": census_summary.get("policy_seed_offset"),
        "exclusions": exclusions,
        "excluded_union_count": len(excluded),
        "enrollment_rule": enrollment_rule,
        "cohorts": {
            "anchor_train": exclusions.get("anchor_train", []),
            "anchor_probe": exclusions.get("anchor_probe", []),
            "untouched_preservation": untouched,
            "boundary": boundary,
            "id_heldout": all_id,
            "train_seen": all_train,
        },
        "counts": {
            "untouched_preservation": len(untouched),
            "boundary": len(boundary),
            "anchor_train": len(exclusions.get("anchor_train", [])),
            "anchor_probe": len(exclusions.get("anchor_probe", [])),
        },
        "meets_min_untouched": len(untouched) >= min_untouched,
        "min_untouched_target": min_untouched,
        "feasibility_by_rule": feasibility_projection(
            id_counts,
            excluded=excluded,
            min_untouched=min_untouched,
            repeats_required=repeats_required,
        ),
        "primary_endpoint_split": "untouched_preservation",
        "base_eval": str(base_eval),
        "seeds_file": str(seeds_file),
        "selection_rule": {
            "untouched_preservation": (
                f"integer success_count >= {min_successes} of {repeats_required} "
                f"on census {enrollment_split}"
            ),
            "boundary": "eligible train_seen seeds ranked by base success-rate distance to 0.5, then env_seed",
        },
        "boundary_base_success_rates": {str(seed): train_rates[seed] for seed in boundary},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="place_container_plate")
    parser.add_argument("--census-dir", type=Path, help="Directory containing census_summary.json")
    parser.add_argument("--census-summary", type=Path, help="Explicit census_summary.json path")
    parser.add_argument(
        "--calibration-run-dir",
        type=Path,
        default=BRACE_DIR / "runs/20260803T031036Z_anchor_calibration_place_container_plate",
    )
    parser.add_argument(
        "--behavior-eval-dir",
        type=Path,
        default=BRACE_DIR / "runs/20260803T073015Z_anchor_behavior_eval_place_container_plate_dump_bin_bigbin",
    )
    parser.add_argument("--seeds-file", type=Path, default=BRACE_DIR / "seeds/place_container_plate_confirmatory_v1.4.2_seeds.json")
    parser.add_argument("--pilot-seeds-file", type=Path, default=BRACE_DIR / "seeds/place_container_plate_pilot_seeds.json")
    parser.add_argument("--confirm-seeds-file", type=Path, default=BRACE_DIR / "seeds/place_container_plate_confirm_seeds.json")
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=BRACE_DIR / "screen_protocol.v1.4.2.confirmatory_preservation.json",
    )
    parser.add_argument("--min-untouched", type=int, default=60)
    parser.add_argument("--boundary-count", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.dataset_manifest is None:
        dataset_meta = repo_path(args.calibration_run_dir) / "A1" / f"{args.task}_N1.zarr" / "brace_dataset_manifest.json"
        if dataset_meta.is_file():
            source_manifest = Path(str(read_json(dataset_meta).get("source_manifest", "")))
            if source_manifest.is_file():
                args.dataset_manifest = source_manifest
    if args.dataset_manifest is None or not repo_path(args.dataset_manifest).is_file():
        raise SystemExit(
            "missing SFT source manifest; pass --dataset-manifest so confirmatory cohorts can exclude learned chunk seeds"
        )

    protocol_path = repo_path(args.protocol)
    protocol = read_json(protocol_path)
    if protocol.get("status") != "frozen" or bool(protocol.get("exploratory", True)):
        raise SystemExit(f"confirmatory cohort requires a frozen non-exploratory protocol: {protocol_path}")
    frozen_min_untouched = int(protocol["preservation_cohorts"]["min_untouched_base_solved"])
    if args.min_untouched != frozen_min_untouched:
        raise SystemExit(
            f"--min-untouched={args.min_untouched} differs from frozen protocol value {frozen_min_untouched}"
        )

    if args.census_summary:
        census_summary = read_json(repo_path(args.census_summary))
        base_eval = Path(census_summary["eval_path"])
    elif args.census_dir:
        census_summary, base_eval = load_census_artifacts(repo_path(args.census_dir))
    else:
        raise SystemExit("pass --census-dir or --census-summary from P1a confirmatory-base-census")

    validate_census_summary(census_summary, protocol_path=protocol_path, protocol=protocol)
    seeds_file = repo_path(args.seeds_file)
    if file_sha256(seeds_file) != census_summary.get("seeds_file_sha256"):
        raise SystemExit("--seeds-file does not match the seed file frozen by the census summary")
    behavior_eval_dir = repo_path(args.behavior_eval_dir) if args.behavior_eval_dir else None
    if behavior_eval_dir is None or not behavior_eval_dir.is_dir():
        raise SystemExit("missing Phase 3C behavior eval directory required by exclude_sources")
    enrollment_rule = protocol["base_solved_enrollment"]

    exclusions = collect_exclusions(
        task=args.task,
        calibration_run_dir=repo_path(args.calibration_run_dir),
        behavior_eval_dir=behavior_eval_dir,
        pilot_seeds_file=repo_path(args.pilot_seeds_file) if args.pilot_seeds_file else None,
        confirm_seeds_file=repo_path(args.confirm_seeds_file) if args.confirm_seeds_file else None,
        dataset_manifest=repo_path(args.dataset_manifest) if args.dataset_manifest else None,
    )
    payload = select_preservation_cohort(
        task=args.task,
        base_eval=base_eval,
        census_summary=census_summary,
        seeds_file=seeds_file,
        exclusions=exclusions,
        enrollment_rule=enrollment_rule,
        min_untouched=args.min_untouched,
        boundary_count=args.boundary_count,
    )
    payload["protocol_path"] = str(protocol_path)
    payload["protocol_sha256"] = file_sha256(protocol_path)
    census_summary_path = (
        repo_path(args.census_summary)
        if args.census_summary
        else repo_path(args.census_dir) / "census_summary.json"
    )
    payload["census_summary_path"] = str(census_summary_path)
    payload["input_sha256"] = {
        "census_summary": file_sha256(census_summary_path),
        "census_eval": file_sha256(base_eval),
        "seeds_file": file_sha256(repo_path(args.seeds_file)),
        "dataset_manifest": file_sha256(repo_path(args.dataset_manifest)),
    }
    payload["seed_set_sha256"] = seed_set_sha256(payload["cohorts"]["id_heldout"] + payload["cohorts"]["train_seen"])
    output = repo_path(args.output)
    write_json_atomic(output, payload)
    print(json.dumps(payload, indent=2))
    return 0 if payload["meets_min_untouched"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
