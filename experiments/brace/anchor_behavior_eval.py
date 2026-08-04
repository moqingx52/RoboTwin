#!/usr/bin/env python3
"""Phase 3C behavior evaluation for anchor calibration checkpoints."""

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

from experiments.brace.replay_audit import read_json, write_json_atomic
from experiments.brace.screen_gates import compute_forgetting_gate, episode_outcomes


def base_checkpoint(task: str) -> Path:
    return DP_DIR / "checkpoints" / f"{task}-demo_clean-200-0" / "600.ckpt"


def eval_output_path(output_dir: Path, task: str, variant: str) -> Path:
    return output_dir / task / f"{variant}.json"


def find_anchor_manifest(calibration_run_dir: Path, *, task: str = "place_container_plate") -> Path:
    for job_dir in sorted(calibration_run_dir.iterdir()):
        if not job_dir.is_dir() or not job_dir.name.startswith("A"):
            continue
        manifest = job_dir / f"{task}_anchor_replay.zarr" / "brace_anchor_manifest.json"
        if manifest.is_file():
            return manifest
    raise FileNotFoundError(f"no anchor manifest under {calibration_run_dir}")


def base_solved_seeds_from_manifest(manifest_path: Path) -> list[int]:
    manifest = read_json(manifest_path)
    return sorted(
        {
            int(row["env_seed"])
            for row in manifest.get("episodes", [])
            if str(row.get("preservation_group")) == "base_solved"
        }
    )


def audit_base_solved_coverage(
    *,
    base_solved_seeds: list[int],
    seeds_file: Path,
    train_seed_count: int = 20,
) -> dict[str, Any]:
    payload = read_json(seeds_file)
    train_seen = [int(seed) for seed in payload["train_rollout"][:train_seed_count]]
    base_set = set(base_solved_seeds)
    train_set = set(train_seen)
    missing = sorted(base_set - train_set)
    return {
        "base_solved_seed_count": len(base_solved_seeds),
        "train_seen_count": len(train_seen),
        "train_seen_overlap": len(base_set & train_set),
        "train_seen_missing": missing,
        "requires_base_solved_anchor_split": bool(missing),
    }


def build_extra_splits_file(
    *,
    base_solved_seeds: list[int],
    audit: dict[str, Any],
    work_dir: Path,
) -> Path | None:
    if not audit["requires_base_solved_anchor_split"]:
        return None
    path = work_dir / "base_solved_anchor_split.json"
    write_json_atomic(path, {"base_solved_anchor": base_solved_seeds})
    return path


def inspect_diagnostic_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    try:
        import dill
        import torch
        from omegaconf import OmegaConf
    except ImportError as exc:
        return {"checkpoint_path": str(checkpoint_path), "inspect_error": str(exc)}

    payload = torch.load(checkpoint_path.open("rb"), pickle_module=dill, map_location="cpu")
    cfg = payload.get("cfg")
    state_dicts = payload.get("state_dicts", {})
    use_ema = bool(OmegaConf.select(cfg, "training.use_ema", default=False)) if cfg is not None else None
    return {
        "checkpoint_path": str(checkpoint_path),
        "has_ema_model": "ema_model" in state_dicts,
        "has_model": "model" in state_dicts,
        "use_ema_cfg": use_ema,
        "deploys_ema": bool(use_ema and "ema_model" in state_dicts),
        "state_dict_keys": sorted(state_dicts.keys()),
    }


def checkpoint_probe_fields(job_summary: dict[str, Any], step: int | str) -> dict[str, Any]:
    """Return checkpoint-specific probe fields; never fall back to run-final probe_summary."""
    step_key = str(step)
    artifacts = job_summary.get("checkpoint_artifacts")
    if not isinstance(artifacts, dict) or step_key not in artifacts:
        raise KeyError(f"missing checkpoint_artifacts[{step_key!r}]")
    artifact = artifacts[step_key]
    if not isinstance(artifact, dict) or not isinstance(artifact.get("probe_constraints"), dict):
        raise ValueError(f"checkpoint_artifacts[{step_key!r}] has no probe_constraints")
    probe_constraints = artifact.get("probe_constraints", {})
    probe_monitor = artifact.get("probe_monitor", {})
    return {
        "probe_constraints": probe_constraints,
        "probe_summary": {
            "base_solved": {
                "probe_tail_mean": probe_constraints.get("base_solved"),
                "probe_tail_ema_mean": probe_monitor.get("base_solved_ema_drift"),
            },
            "boundary": {
                "probe_tail_mean": probe_constraints.get("boundary"),
                "probe_tail_ema_mean": probe_monitor.get("boundary_ema_drift"),
            },
            "rows": int(step_key) if step_key.isdigit() else None,
            "tail_mode": "checkpoint_artifact",
            "tail_steps": 50,
            "source": "checkpoint_artifacts",
        },
        "checkpoint_artifact": artifact,
    }


def pick_best_formulation_job(calibration_run_dir: Path, job_ids: tuple[str, ...] = ("A5", "A6", "A7")) -> str | None:
    best_job = None
    best_score = float("inf")
    for job_id in job_ids:
        summary_path = calibration_run_dir / job_id / "summary.json"
        if not summary_path.is_file():
            continue
        summary = read_json(summary_path)
        if not summary.get("complete"):
            continue
        probe = summary.get("probe_summary", {})
        base_stats = probe.get("base_solved", {})
        boundary_stats = probe.get("boundary", {})
        base = base_stats.get("probe_tail_ema_mean", base_stats.get("probe_tail_mean"))
        boundary = boundary_stats.get("probe_tail_ema_mean", boundary_stats.get("probe_tail_mean"))
        if base is None or boundary is None:
            continue
        score = float(base) + float(boundary)
        if score < best_score:
            best_score = score
            best_job = job_id
    return best_job


def build_recommended_candidates(calibration_run_dir: Path) -> list[dict[str, Any]]:
    best_formulation = pick_best_formulation_job(calibration_run_dir)
    specs: list[dict[str, Any]] = [
        {"source": "base", "label": "base_original"},
        {"job_id": "A0", "step": 1235, "label": "sft_only"},
        {"job_id": "A1", "step": 1235, "label": "dual_mid"},
        {"job_id": "A1", "step": 2470, "label": "dual_low_drift"},
        {"job_id": "A3", "step": 1235, "label": "high_dual_lr"},
    ]
    if best_formulation is not None:
        specs.append(
            {
                "job_id": best_formulation,
                "step": 1235,
                "label": f"best_formulation_{best_formulation.lower()}",
            }
        )
    return specs


def resolve_checkpoint_path(calibration_run_dir: Path, spec: dict[str, Any], task: str) -> str | None:
    if spec.get("source") == "base":
        ckpt = base_checkpoint(task)
        return str(ckpt) if ckpt.is_file() else None
    job_id = spec["job_id"]
    step = int(spec["step"])
    job_dir = calibration_run_dir / job_id
    summary_path = job_dir / "summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        ckpt = summary.get("checkpoint_paths", {}).get(str(step))
        if ckpt and Path(ckpt).is_file():
            return str(ckpt)
    ckpt_path = job_dir / "checkpoints" / f"step_{step}.ckpt"
    return str(ckpt_path) if ckpt_path.is_file() else None


def select_checkpoint_candidates(
    calibration_run_dir: Path,
    *,
    task: str,
    candidates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in candidates or build_recommended_candidates(calibration_run_dir):
        ckpt = resolve_checkpoint_path(calibration_run_dir, spec, task)
        if not ckpt:
            continue
        row = {
            "label": spec.get("label", spec.get("job_id", "base")),
            "checkpoint_path": ckpt,
            "source": spec.get("source", "calibration"),
            "job_id": spec.get("job_id"),
            "step": spec.get("step"),
        }
        if spec.get("job_id"):
            job_dir = calibration_run_dir / spec["job_id"]
            summary_path = job_dir / "summary.json"
            if summary_path.is_file():
                summary = read_json(summary_path)
                step = spec.get("step")
                if step is not None:
                    row.update(checkpoint_probe_fields(summary, step))
                else:
                    row["probe_summary"] = summary.get("probe_summary")
        row["checkpoint_audit"] = inspect_diagnostic_checkpoint(Path(ckpt))
        rows.append(row)
    return rows


def validate_recommended_candidates(candidates: list[dict[str, Any]]) -> None:
    """Fail closed before scheduling an incomplete or non-deployable candidate set."""
    expected_labels = {
        "base_original",
        "sft_only",
        "dual_mid",
        "dual_low_drift",
        "high_dual_lr",
    }
    labels = [str(row.get("label")) for row in candidates]
    missing = sorted(expected_labels - set(labels))
    formulation_labels = [label for label in labels if label.startswith("best_formulation_")]
    if missing or len(formulation_labels) != 1 or len(labels) != 6 or len(set(labels)) != len(labels):
        raise RuntimeError(
            "expected six unique behavior candidates "
            f"(base/A0/A1@1235/A1@2470/A3/best formulation); labels={labels}, missing={missing}"
        )

    non_ema = [
        row["label"]
        for row in candidates
        if not bool(row.get("checkpoint_audit", {}).get("deploys_ema"))
    ]
    if non_ema:
        raise RuntimeError(f"behavior candidates do not deploy EMA: {non_ema}")


def build_eval_command(
    *,
    task: str,
    variant: str,
    checkpoint_path: str,
    output_dir: Path,
    seeds_file: Path,
    hard_seeds_file: Path,
    extra_splits_file: Path | None,
    workers_per_gpu: int,
    protocol: dict[str, Any] | None = None,
    eval_profile: str = "confirmatory",
) -> list[str]:
    if protocol is None:
        eval_cfg: dict[str, Any] = {}
    elif eval_profile == "census":
        eval_cfg = protocol.get("census_eval", {})
    else:
        eval_cfg = protocol.get("eval", {})
    include_hard = int(eval_cfg.get("hard_seed_count", 20)) > 0 and int(eval_cfg.get("hard_repeats", 0)) > 0
    census_candidate_split = eval_profile == "census" and "census_candidate_id_count" in eval_cfg
    command = [
        "python",
        str(BRACE_DIR / "run_eval_group.py"),
        "--workers",
        str(workers_per_gpu),
        "--",
        "python",
        str(PHASE1_DIR / "eval_per_seed.py"),
        "--task",
        task,
        "--task-config",
        "demo_clean",
        "--variant",
        variant,
        "--ckpt-path",
        checkpoint_path,
        "--output-dir",
        str(output_dir),
        "--seeds-file",
        str(seeds_file),
        "--hard-seeds-file",
        str(hard_seeds_file),
    ]
    if census_candidate_split:
        command.extend(
            [
                "--census-candidate-split",
                "--census-candidate-id-count",
                str(eval_cfg["census_candidate_id_count"]),
            ]
        )
    else:
        command.extend(
            [
                "--id-seed-count",
                str(eval_cfg.get("id_seed_count", 20)),
            ]
        )
    command.extend(
        [
        "--train-seed-count",
        str(eval_cfg.get("train_seed_count", 20)),
        "--hard-seed-count",
        str(eval_cfg.get("hard_seed_count", 20)),
        "--id-repeats",
        str(eval_cfg.get("id_repeats", 1)),
        "--train-repeats",
        str(eval_cfg.get("train_repeats", 1)),
        "--hard-repeats",
        str(eval_cfg.get("hard_repeats", 1)),
        "--extra-split-repeats",
        str(eval_cfg.get("extra_split_repeats", eval_cfg.get("preservation_repeats", eval_cfg.get("id_repeats", 1)))),
        "--policy-seed-offset",
        str(eval_cfg.get("policy_seed_offset", 2000)),
        "--resume",
        ]
    )
    if not include_hard:
        command.append("--no-include-hard")
    if extra_splits_file is not None:
        command.extend(["--extra-splits-file", str(extra_splits_file)])
    return command


def aggregate_preservation_metrics(
    *,
    base_eval_path: Path,
    candidate_eval_path: Path,
    base_solved_seeds: set[int],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    split = "base_solved_anchor"
    if not episode_outcomes(candidate_eval_path, split):
        split = "train_seen"
    base_outcomes = episode_outcomes(base_eval_path, split)
    candidate_outcomes = episode_outcomes(candidate_eval_path, split)
    paired = [seed for seed in base_solved_seeds if seed in base_outcomes and seed in candidate_outcomes]
    base_success_seeds = [seed for seed in paired if base_outcomes[seed]]
    forgetting = []
    for seed in paired:
        if base_outcomes[seed] and not candidate_outcomes[seed]:
            forgetting.append(seed)
    return {
        "split": split,
        "paired_seeds": len(paired),
        "base_success_count": len(base_success_seeds),
        "forgetting_seeds": forgetting,
        "forgetting_rate": (
            len(forgetting) / len(base_success_seeds) if base_success_seeds else None
        ),
        "candidate_success_rate": (
            sum(candidate_outcomes[seed] for seed in paired) / len(paired) if paired else None
        ),
        "base_success_rate": (
            sum(base_outcomes[seed] for seed in paired) / len(paired) if paired else None
        ),
    }


def run_single_candidate_eval(
    candidate: dict[str, Any],
    *,
    task: str,
    output_dir: Path,
    seeds_file: Path,
    hard_seeds_file: Path,
    extra_splits_file: Path | None,
    workers_per_gpu: int,
    gpu: int,
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    label = candidate["label"]
    variant = f"calib_{label}"
    command = build_eval_command(
        task=task,
        variant=variant,
        checkpoint_path=candidate["checkpoint_path"],
        output_dir=output_dir,
        seeds_file=seeds_file,
        hard_seeds_file=hard_seeds_file,
        extra_splits_file=extra_splits_file,
        workers_per_gpu=workers_per_gpu,
        protocol=protocol,
    )
    env = dict(**__import__("os").environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    eval_path = eval_output_path(output_dir, task, variant)
    return {
        **candidate,
        "variant": variant,
        "eval_output": str(eval_path),
        "eval_deploys_ema": candidate.get("checkpoint_audit", {}).get("deploys_ema"),
    }


def run_behavior_eval(
    *,
    task: str,
    calibration_run_dir: Path,
    output_dir: Path,
    protocol: dict[str, Any] | None = None,
    candidates: list[dict[str, Any]] | None = None,
    seeds_file: Path | None = None,
    workers_per_gpu: int = 3,
    gpu: int = 0,
    single_label: str | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = protocol or {}
    seeds_file = seeds_file or (PHASE1_DIR / "seeds" / f"{task}_seeds.json")
    hard_seeds_file = PHASE1_DIR / "eval_results_200" / "hard_eval_seeds" / f"{task}.json"

    manifest_path = find_anchor_manifest(calibration_run_dir, task=task)
    base_solved_seeds = base_solved_seeds_from_manifest(manifest_path)
    coverage_audit_path = output_dir / "base_solved_coverage_audit.json"
    if coverage_audit_path.is_file():
        coverage_audit = read_json(coverage_audit_path)
    else:
        coverage_audit = audit_base_solved_coverage(
            base_solved_seeds=base_solved_seeds,
            seeds_file=seeds_file,
        )
        write_json_atomic(coverage_audit_path, coverage_audit)
    extra_splits_path = output_dir / "base_solved_anchor_split.json"
    extra_splits_file = extra_splits_path if extra_splits_path.is_file() else build_extra_splits_file(
        base_solved_seeds=base_solved_seeds,
        audit=coverage_audit,
        work_dir=output_dir,
    )

    selected = select_checkpoint_candidates(calibration_run_dir, task=task, candidates=candidates)
    if candidates is None:
        validate_recommended_candidates(selected)
    if single_label is not None:
        selected = [row for row in selected if row["label"] == single_label]
    if not selected:
        raise RuntimeError(f"no calibration checkpoints found under {calibration_run_dir}")

    eval_results = [
        run_single_candidate_eval(
            row,
            task=task,
            output_dir=output_dir,
            seeds_file=seeds_file,
            hard_seeds_file=hard_seeds_file,
            extra_splits_file=extra_splits_file,
            workers_per_gpu=workers_per_gpu,
            gpu=gpu,
            protocol=protocol,
        )
        for row in selected
    ]

    if single_label is not None:
        partial = eval_results[0]
        partial_path = output_dir / "partials" / f"{single_label}.json"
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(partial_path, partial)
        return partial

    return finalize_behavior_eval_summary(
        task=task,
        calibration_run_dir=calibration_run_dir,
        output_dir=output_dir,
        protocol=protocol,
        seeds_file=seeds_file,
        base_solved_seeds=base_solved_seeds,
        coverage_audit=coverage_audit,
        extra_splits_file=extra_splits_file,
        eval_results=eval_results,
    )


def finalize_behavior_eval_summary(
    *,
    task: str,
    calibration_run_dir: Path,
    output_dir: Path,
    protocol: dict[str, Any],
    seeds_file: Path,
    base_solved_seeds: list[int],
    coverage_audit: dict[str, Any],
    extra_splits_file: Path | None,
    eval_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if eval_results is None:
        partial_dir = output_dir / "partials"
        eval_results = [read_json(path) for path in sorted(partial_dir.glob("*.json"))] if partial_dir.is_dir() else []

    base_eval = next((row for row in eval_results if row["label"] == "base_original"), None)
    base_solved_set = set(base_solved_seeds)
    for row in eval_results:
        if base_eval is None:
            row["preservation"] = {"error": "missing_base_original_eval"}
            continue
        row["preservation"] = aggregate_preservation_metrics(
            base_eval_path=Path(base_eval["eval_output"]),
            candidate_eval_path=Path(row["eval_output"]),
            base_solved_seeds=base_solved_set,
            protocol=protocol,
        )

    forgetting_gate = None
    if base_eval is not None and protocol:
        sft_eval = next((row for row in eval_results if row["label"] == "sft_only"), None)
        dual_eval = next((row for row in eval_results if row["label"] == "dual_low_drift"), None)
        if sft_eval and dual_eval:
            forgetting_gate = compute_forgetting_gate(
                base_eval=Path(base_eval["eval_output"]),
                u1_eval=Path(sft_eval["eval_output"]),
                b2_eval=Path(dual_eval["eval_output"]),
                base_solved_seeds=base_solved_set,
                protocol=protocol,
            )

    summary = {
        "schema_version": 2,
        "stage": "anchor_behavior_eval",
        "task": task,
        "calibration_run_dir": str(calibration_run_dir),
        "seeds_file": str(seeds_file),
        "base_solved_coverage_audit": coverage_audit,
        "extra_splits_file": str(extra_splits_file) if extra_splits_file else None,
        "eval_note": (
            "DP deploy path uses workspace.ema_model when cfg.training.use_ema=true; "
            "diagnostic checkpoints include state_dicts.ema_model."
        ),
        "forgetting_gate_dual_vs_sft": forgetting_gate,
        "candidates": eval_results,
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def repair_behavior_eval_probe_links(
    *,
    summary_path: Path,
    calibration_run_dir: Path,
    seal: bool = True,
) -> dict[str, Any]:
    summary = read_json(summary_path)
    repaired: list[str] = []
    expected: list[str] = []
    for row in summary.get("candidates", []):
        job_id = row.get("job_id")
        step = row.get("step")
        if not job_id or step is None:
            continue
        label = str(row.get("label", f"{job_id}@{step}"))
        expected.append(label)
        job_summary_path = calibration_run_dir / str(job_id) / "summary.json"
        if not job_summary_path.is_file():
            raise FileNotFoundError(f"missing calibration summary for {label}: {job_summary_path}")
        row.update(checkpoint_probe_fields(read_json(job_summary_path), step))
        repaired.append(label)
    if repaired != expected:
        raise RuntimeError(f"incomplete probe repair: expected={expected}, repaired={repaired}")
    summary["probe_linkage_version"] = "checkpoint_artifacts_v1"
    summary["probe_linkage_candidates"] = repaired
    if seal:
        summary["sealed"] = True
        summary["seal_note"] = (
            "Probe metrics joined from per-checkpoint checkpoint_artifacts; "
            "do not use run-final probe_summary for drift-behavior plots."
        )
    write_json_atomic(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="place_container_plate")
    parser.add_argument("--calibration-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=BRACE_DIR / "screen_protocol.v1.3.exploratory_calibration.json")
    parser.add_argument("--candidates-json", type=Path, help="Optional JSON list of candidate specs")
    parser.add_argument("--seeds-file", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument("--label", help="Run a single candidate label from the recommended set")
    parser.add_argument("--aggregate-only", action="store_true", help="Merge partial candidate results into summary.json")
    parser.add_argument("--repair-probe-links", action="store_true", help="Rewrite summary probe fields from checkpoint_artifacts")
    parser.add_argument("--summary-path", type=Path, help="Existing behavior-eval summary to repair (with --repair-probe-links)")
    args = parser.parse_args()

    if args.repair_probe_links:
        summary_path = (args.summary_path or args.output.resolve() / "summary.json").resolve()
        summary = repair_behavior_eval_probe_links(
            summary_path=summary_path,
            calibration_run_dir=args.calibration_run_dir.resolve(),
        )
        print(json.dumps(summary, indent=2))
        return 0

    protocol = read_json(args.protocol.resolve()) if args.protocol.is_file() else {}
    candidates = None
    if args.candidates_json:
        candidates = read_json(args.candidates_json.resolve())
    elif args.label:
        candidates = [
            spec
            for spec in build_recommended_candidates(args.calibration_run_dir.resolve())
            if spec.get("label") == args.label
        ]

    if args.aggregate_only:
        manifest_path = find_anchor_manifest(args.calibration_run_dir.resolve(), task=args.task)
        base_solved_seeds = base_solved_seeds_from_manifest(manifest_path)
        coverage_audit = read_json(args.output.resolve() / "base_solved_coverage_audit.json")
        extra_splits = args.output.resolve() / "base_solved_anchor_split.json"
        summary = finalize_behavior_eval_summary(
            task=args.task,
            calibration_run_dir=args.calibration_run_dir.resolve(),
            output_dir=args.output.resolve(),
            protocol=protocol,
            seeds_file=(args.seeds_file.resolve() if args.seeds_file else PHASE1_DIR / "seeds" / f"{args.task}_seeds.json"),
            base_solved_seeds=base_solved_seeds,
            coverage_audit=coverage_audit,
            extra_splits_file=extra_splits if extra_splits.is_file() else None,
        )
        print(json.dumps(summary, indent=2))
        return 0

    summary = run_behavior_eval(
        task=args.task,
        calibration_run_dir=args.calibration_run_dir.resolve(),
        output_dir=args.output.resolve(),
        protocol=protocol,
        candidates=candidates,
        seeds_file=args.seeds_file.resolve() if args.seeds_file else None,
        workers_per_gpu=args.workers_per_gpu,
        gpu=args.gpu,
        single_label=args.label,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
