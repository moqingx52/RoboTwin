#!/usr/bin/env python3
"""Run confirmatory paired eval for base + C0/C1 training seeds."""

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

from experiments.brace.anchor_behavior_eval import base_checkpoint, build_eval_command, inspect_diagnostic_checkpoint
from experiments.brace.confirmatory_common import file_sha256, seed_set_sha256
from experiments.brace.replay_audit import read_json, write_json_atomic


def cohort_extra_splits(cohort: dict[str, Any]) -> dict[str, list[int]]:
    splits = {
        key: [int(seed) for seed in values]
        for key, values in cohort.get("cohorts", {}).items()
        if key in {"untouched_preservation", "anchor_train", "anchor_probe", "boundary"}
    }
    return splits


def resolve_checkpoint(
    *,
    label: str,
    training_run_dir: Path | None,
    task: str,
    primary_step: int,
    protocol: dict[str, Any],
    protocol_sha256: str,
) -> dict[str, Any]:
    if label == "base_original":
        ckpt = base_checkpoint(task)
        audit = inspect_diagnostic_checkpoint(ckpt)
        if not audit.get("deploys_ema"):
            raise RuntimeError(f"base checkpoint does not deploy EMA: {ckpt}")
        return {
            "label": label,
            "checkpoint_path": str(ckpt),
            "checkpoint_sha256": file_sha256(ckpt),
            "checkpoint_audit": audit,
        }
    if training_run_dir is None:
        raise RuntimeError("training_run_dir required for training checkpoints")
    if not (label.startswith("c0_s") or label.startswith("c1_s")):
        raise ValueError(f"unsupported confirmatory label: {label}")
    method_prefix, seed_text = label.split("_s", 1)
    if not seed_text.isdigit():
        raise ValueError(f"invalid confirmatory label: {label}")
    job_id = f"C0_s{seed_text}" if method_prefix == "c0" else f"C1_s{seed_text}"
    summary_path = training_run_dir / job_id / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = read_json(summary_path)
    if not summary.get("complete"):
        raise RuntimeError(f"training job is incomplete: {job_id}")
    if int(summary.get("training_seed", -1)) != int(seed_text):
        raise RuntimeError(f"training seed mismatch for {job_id}: {summary.get('training_seed')!r}")
    if summary.get("protocol_sha256") != protocol_sha256:
        raise RuntimeError(f"training protocol SHA mismatch for {job_id}")
    if summary.get("job_id") != job_id:
        raise RuntimeError(f"training job_id mismatch for {job_id}")
    jobs_manifest_path = Path(str(summary.get("jobs_manifest_path", "")))
    if (
        not jobs_manifest_path.is_file()
        or summary.get("jobs_manifest_sha256") != file_sha256(jobs_manifest_path)
    ):
        raise RuntimeError(f"training jobs manifest provenance mismatch for {job_id}")
    step_key = str(primary_step)
    ckpt = summary.get("checkpoint_paths", {}).get(step_key)
    if not ckpt or not Path(ckpt).is_file():
        ckpt_path = training_run_dir / job_id / "checkpoints" / f"step_{primary_step}.ckpt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"missing checkpoint for {job_id} step {primary_step}")
        ckpt = str(ckpt_path)
    audit = inspect_diagnostic_checkpoint(Path(ckpt))
    if not audit.get("deploys_ema"):
        raise RuntimeError(f"{label} checkpoint does not deploy EMA: {ckpt}")
    operational_probe_gate = None
    if method_prefix == "c1":
        artifact = summary.get("checkpoint_artifacts", {}).get(step_key, {})
        probe_stats = artifact.get("probe_stats", {})
        budget = protocol.get("constraint_budget", {}).get("operational_probe_budget", {})
        observed = {
            "base_solved_tail_p90_max": probe_stats.get("probe_constraint_p90/base_solved"),
            "boundary_tail_p90_max": probe_stats.get("probe_constraint_p90/boundary"),
        }
        checks = {
            key: observed.get(key) is not None and float(observed[key]) <= float(limit)
            for key, limit in budget.items()
        }
        operational_probe_gate = {
            "passed": bool(checks) and all(checks.values()),
            "observed": observed,
            "limits": budget,
            "checks": checks,
        }
    return {
        "label": label,
        "job_id": job_id,
        "checkpoint_path": ckpt,
        "checkpoint_sha256": file_sha256(Path(ckpt)),
        "checkpoint_audit": audit,
        "training_seed": summary.get("training_seed"),
        "operational_probe_gate": operational_probe_gate,
    }


def run_single_eval(
    *,
    label: str,
    task: str,
    output_dir: Path,
    protocol: dict[str, Any],
    seeds_file: Path,
    hard_seeds_file: Path,
    extra_splits: dict[str, list[int]],
    workers_per_gpu: int,
    gpu: int | None,
    training_run_dir: Path | None,
    protocol_sha256: str,
    cohort_sha256: str,
) -> dict[str, Any]:
    primary_step = int(protocol.get("checkpoint_primary_step", 2470))
    candidate = resolve_checkpoint(
        label=label,
        training_run_dir=training_run_dir,
        task=task,
        primary_step=primary_step,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_splits_path = output_dir / f"extra_splits_{label}.json"
    write_json_atomic(extra_splits_path, extra_splits)
    variant = "confirm_base_original" if label == "base_original" else f"confirm_{label}"
    command = build_eval_command(
        task=task,
        variant=variant,
        checkpoint_path=candidate["checkpoint_path"],
        output_dir=output_dir,
        seeds_file=seeds_file,
        hard_seeds_file=hard_seeds_file,
        extra_splits_file=extra_splits_path,
        workers_per_gpu=workers_per_gpu,
        protocol=protocol,
        eval_profile="confirmatory",
    )
    env = dict(**__import__("os").environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    eval_path = output_dir / task / f"{variant}.json"
    return {
        **candidate,
        "variant": variant,
        "eval_output": str(eval_path),
        "eval_sha256": file_sha256(eval_path),
        "protocol_sha256": protocol_sha256,
        "cohort_sha256": cohort_sha256,
        "policy_seed_offset": int(protocol["eval"]["policy_seed_offset"]),
    }


def build_candidate_labels(protocol: dict[str, Any]) -> list[str]:
    labels = ["base_original"]
    for seed in protocol["training_seeds"]:
        labels.append(f"c0_s{seed}")
        labels.append(f"c1_s{seed}")
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="place_container_plate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=BRACE_DIR / "screen_protocol.v1.4.1.confirmatory_preservation.json")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--training-run-dir", type=Path)
    parser.add_argument("--seeds-file", type=Path)
    parser.add_argument("--hard-seeds-file", type=Path, default=PHASE1_DIR / "eval_results_200/hard_eval_seeds/place_container_plate.json")
    parser.add_argument("--label")
    parser.add_argument(
        "--gpu",
        type=int,
        help="Physical GPU for standalone use; omit under an orchestrator and inherit CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    cohort_path = args.cohort.resolve()
    protocol = read_json(protocol_path)
    cohort = read_json(cohort_path)
    if protocol.get("status") != "frozen" or bool(protocol.get("exploratory", True)):
        raise SystemExit("confirmatory eval requires a frozen non-exploratory protocol")
    if args.task not in protocol.get("tasks", []):
        raise SystemExit(f"task {args.task!r} is not frozen in the confirmatory protocol")
    protocol_sha256 = file_sha256(protocol_path)
    cohort_sha256 = file_sha256(cohort_path)
    if not cohort.get("frozen") or not cohort.get("meets_min_untouched"):
        raise SystemExit("confirmatory eval requires a frozen cohort meeting min_untouched")
    if cohort.get("protocol_sha256") != protocol_sha256:
        raise SystemExit("cohort protocol SHA does not match confirmatory protocol")
    if int(cohort.get("census_policy_seed_offset", -1)) == int(protocol["eval"]["policy_seed_offset"]):
        raise SystemExit("census and confirmatory policy seed offsets must differ")
    seeds_file = (args.seeds_file or (PHASE1_DIR / f"seeds/{args.task}_seeds.json")).resolve()
    if cohort.get("input_sha256", {}).get("seeds_file") != file_sha256(seeds_file):
        raise SystemExit("eval seeds file does not match cohort provenance")
    expected_seed_set_sha = seed_set_sha256(
        [int(seed) for seed in cohort["cohorts"]["id_heldout"]]
        + [int(seed) for seed in cohort["cohorts"]["train_seen"]]
    )
    if cohort.get("seed_set_sha256") != expected_seed_set_sha:
        raise SystemExit("cohort seed_set_sha256 mismatch")
    extra_splits = cohort_extra_splits(cohort)
    labels = [args.label] if args.label else build_candidate_labels(protocol)
    results = []
    for label in labels:
        row = run_single_eval(
            label=label,
            task=args.task,
            output_dir=args.output.resolve(),
            protocol=protocol,
            seeds_file=seeds_file.resolve(),
            hard_seeds_file=args.hard_seeds_file.resolve(),
            extra_splits=extra_splits,
            workers_per_gpu=args.workers_per_gpu,
            gpu=args.gpu,
            training_run_dir=args.training_run_dir.resolve() if args.training_run_dir else None,
            protocol_sha256=protocol_sha256,
            cohort_sha256=cohort_sha256,
        )
        partial_path = args.output.resolve() / "partials" / f"{label}.json"
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(partial_path, row)
        results.append(row)
    print(json.dumps(results if len(results) > 1 else results[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
