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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.replay_audit import read_json, write_json_atomic


DEFAULT_CANDIDATES = [
    {"job_id": "A1", "step": 1235, "label": "current_control"},
    {"job_id": "A1", "step": 2470, "label": "current_control_epoch10"},
    {"job_id": "A0", "step": 1235, "label": "sft_only"},
    {"job_id": "A3", "step": 1235, "label": "high_dual_lr"},
]


def select_checkpoint_candidates(
    calibration_run_dir: Path,
    *,
    candidates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in candidates or DEFAULT_CANDIDATES:
        job_id = spec["job_id"]
        step = int(spec["step"])
        job_dir = calibration_run_dir / job_id
        summary_path = job_dir / "summary.json"
        if not summary_path.is_file():
            continue
        summary = read_json(summary_path)
        ckpt_paths = summary.get("checkpoint_paths", {})
        ckpt = ckpt_paths.get(str(step))
        if not ckpt or not Path(ckpt).is_file():
            ckpt_path = job_dir / "checkpoints" / f"step_{step}.ckpt"
            if not ckpt_path.is_file():
                continue
            ckpt = str(ckpt_path)
        artifact = summary.get("checkpoint_artifacts", {}).get(str(step), {})
        rows.append(
            {
                "job_id": job_id,
                "step": step,
                "label": spec.get("label", f"{job_id}_step{step}"),
                "checkpoint_path": ckpt,
                "probe_constraints": artifact.get("probe_constraints", {}),
                "probe_summary": summary.get("probe_summary"),
            }
        )
    return rows


def run_behavior_eval(
    *,
    task: str,
    calibration_run_dir: Path,
    output_dir: Path,
    candidates: list[dict[str, Any]] | None = None,
    workers_per_gpu: int = 3,
    gpu: int = 0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_checkpoint_candidates(calibration_run_dir, candidates=candidates)
    if not selected:
        raise RuntimeError(f"no calibration checkpoints found under {calibration_run_dir}")

    hard_seeds = PHASE1_DIR / "eval_results_200" / "hard_eval_seeds" / f"{task}.json"
    eval_results: list[dict[str, Any]] = []
    for row in selected:
        label = row["label"]
        variant = f"calib_{label}"
        eval_out = output_dir / f"{label}.json"
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
            row["checkpoint_path"],
            "--output-dir",
            str(output_dir),
            "--hard-seeds-file",
            str(hard_seeds),
            "--id-seed-count",
            "20",
            "--train-seed-count",
            "20",
            "--hard-seed-count",
            "20",
            "--id-repeats",
            "1",
            "--train-repeats",
            "1",
            "--hard-repeats",
            "1",
            "--policy-seed-offset",
            "2000",
            "--resume",
        ]
        env = dict(**{k: v for k, v in __import__("os").environ.items()})
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
        eval_results.append(
            {
                **row,
                "eval_output": str(eval_out),
                "variant": variant,
            }
        )

    summary = {
        "schema_version": 1,
        "stage": "anchor_behavior_eval",
        "task": task,
        "calibration_run_dir": str(calibration_run_dir),
        "candidates": eval_results,
        "note": "Deployed EMA eval wiring is pending DP checkpoint export; current eval uses saved diagnostic checkpoints.",
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="place_container_plate")
    parser.add_argument("--calibration-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates-json", type=Path, help="Optional JSON list of {job_id, step, label}")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    args = parser.parse_args()

    candidates = None
    if args.candidates_json:
        candidates = read_json(args.candidates_json.resolve())

    summary = run_behavior_eval(
        task=args.task,
        calibration_run_dir=args.calibration_run_dir.resolve(),
        output_dir=args.output.resolve(),
        candidates=candidates,
        workers_per_gpu=args.workers_per_gpu,
        gpu=args.gpu,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
