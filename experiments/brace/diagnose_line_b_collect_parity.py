#!/usr/bin/env python3
"""Compare frozen-cohort pre-motion materialization under GPU worker packing.

This is a diagnostic-only runner.  It validates reported failing seeds against
their frozen expert_demo cohorts, adds nearby cohort controls, and executes the
same pre-motion path used by ``script/collect_data.py``:

    setup_demo(real cohort episode index) -> play_once -> plan/check -> save_traj_data

Each task is one logical simulator job.  Jobs are assigned to fixed GPU workers
at ``workers_per_gpu`` density so separate isolated and packed runs can measure
planner/resource-contention effects.  Outputs are immutable timestamped run
directories and never touch manifests or ``data/<task>/demo_clean``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import queue as queue_module
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
SEED_DIR = BRACE_DIR / "seeds" / "multitask_v1"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.multitask_protocol import file_sha256, sha256_sidecar_valid
from experiments.brace.replay_audit import write_json_atomic


DEFAULT_FAILURE_SPECS = (
    "beat_block_hammer=120050",
    "handover_mic=140005",
    "handover_mic=140018",
    "lift_pot=150029",
    "stack_bowls_three=210022",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def parse_failure_specs(values: list[str]) -> dict[str, list[int]]:
    parsed: dict[str, list[int]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid failure seed {value!r}; expected TASK=SEED")
        task, raw_seed = value.split("=", 1)
        task = task.strip()
        if not task:
            raise ValueError(f"invalid empty task in {value!r}")
        seed = int(raw_seed)
        parsed.setdefault(task, [])
        if seed not in parsed[task]:
            parsed[task].append(seed)
    return parsed


def nearby_control_indices(
    cohort: list[int],
    failed_indices: list[int],
    *,
    count: int,
) -> list[int]:
    """Select deterministic nearest non-failure cohort indices."""
    failed = set(failed_indices)
    candidates = [index for index in range(len(cohort)) if index not in failed]
    candidates.sort(key=lambda index: (min(abs(index - failed_index) for failed_index in failed), index))
    return candidates[:count]


def build_task_cases(
    task: str,
    failed_seeds: list[int],
    *,
    controls_per_task: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = SEED_DIR / f"{task}.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing seed manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen" or manifest.get("feasibility", {}).get("passed") is not True:
        raise ValueError(f"{task}: manifest is not frozen with passed feasibility")
    if not sha256_sidecar_valid(manifest_path):
        raise ValueError(f"{task}: manifest SHA256 sidecar is invalid")
    cohort = [int(seed) for seed in manifest.get("cohorts", {}).get("expert_demo", [])]
    if len(cohort) != 50:
        raise ValueError(f"{task}: expected 50 expert_demo seeds, found {len(cohort)}")

    failed_indices: list[int] = []
    cases: list[dict[str, Any]] = []
    for seed in failed_seeds:
        if seed not in cohort:
            raise ValueError(f"{task}: reported failure seed {seed} is not in the frozen cohort")
        index = cohort.index(seed)
        failed_indices.append(index)
        cases.append({"role": "reported_failure", "seed": seed, "episode_idx": index})

    for index in nearby_control_indices(cohort, failed_indices, count=controls_per_task):
        cases.append({"role": "nearby_control", "seed": cohort[index], "episode_idx": index})

    provenance = {
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "feasibility_evidence_path": manifest.get("feasibility", {}).get("evidence_path"),
        "feasibility_evidence_sha256": manifest.get("feasibility", {}).get("evidence_sha256"),
        "cohort_count": len(cohort),
    }
    return provenance, cases


def run_task_job(gpu_id: int, job: dict[str, Any], output_root: str) -> list[dict[str, Any]]:
    """Run all attempts for one task in one persistent simulator process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Import simulator-dependent modules only after pinning the physical GPU.
    from experiments.brace.seed_feasibility import classify_probe_error, load_task_probe_args

    task = str(job["task"])
    task_env, base_args = load_task_probe_args(task, "demo_clean")
    base_args["episode_num"] = 50
    base_args["use_seed"] = True
    base_args["need_plan"] = True
    base_args["render_freq"] = 0
    base_args["collect_data"] = False

    rows: list[dict[str, Any]] = []
    for case in job["cases"]:
        for repeat in range(int(job["repeats"])):
            attempt_dir = Path(output_root) / "artifacts" / task / case["role"] / str(case["seed"]) / f"repeat_{repeat}"
            attempt_dir.mkdir(parents=True, exist_ok=False)
            args = dict(base_args)
            args["save_path"] = str(attempt_dir)
            row: dict[str, Any] = {
                "task": task,
                "role": case["role"],
                "seed": int(case["seed"]),
                "episode_idx": int(case["episode_idx"]),
                "repeat": repeat,
                "gpu_id": int(gpu_id),
                "worker_pid": os.getpid(),
                "started_at": utc_now(),
                "plan_success": False,
                "check_success": False,
                "trajectory_saved": False,
                "passed": False,
                "failed_stage": None,
                "error_type": None,
                "error_message": None,
                "duration_seconds": None,
                "trajectory_path": None,
            }
            started = time.monotonic()
            try:
                task_env.setup_demo(now_ep_num=int(case["episode_idx"]), seed=int(case["seed"]), **args)
                task_env.play_once()
                row["plan_success"] = bool(task_env.plan_success)
                if not row["plan_success"]:
                    row["failed_stage"] = "play_once_plan"
                    row["error_type"] = "expert_plan_failed"
                    row["error_message"] = "expert plan failed during collection-path pre-motion"
                    continue
                row["check_success"] = bool(task_env.check_success())
                if not row["check_success"]:
                    row["failed_stage"] = "check_success"
                    row["error_type"] = "expert_check_failed"
                    row["error_message"] = "expert success check failed during collection-path pre-motion"
                    continue
                task_env.save_traj_data(int(case["episode_idx"]))
                trajectory_path = attempt_dir / "_traj_data" / f"episode{case['episode_idx']}.pkl"
                if not trajectory_path.is_file():
                    raise RuntimeError(f"save_traj_data did not create {trajectory_path}")
                row["trajectory_saved"] = True
                row["trajectory_path"] = str(trajectory_path.relative_to(REPO_ROOT))
                row["passed"] = True
            except Exception as exc:
                if row["failed_stage"] is None:
                    row["failed_stage"] = "exception"
                error_type, message = classify_probe_error(exc)
                row["error_type"] = error_type
                row["error_message"] = message
            finally:
                row["duration_seconds"] = round(time.monotonic() - started, 6)
                rows.append(row)
                task_env.close_env()
    return rows


def worker_main(gpu_id: int, job_queue: Any, result_queue: Any, output_root: str) -> None:
    while True:
        item = job_queue.get()
        if item is None:
            return
        job_index, job = item
        try:
            result_queue.put((job_index, run_task_job(gpu_id, job, output_root), None))
        except Exception as exc:
            result_queue.put((job_index, [], f"{type(exc).__name__}: {exc}"))


def run_jobs_parallel(
    jobs: list[dict[str, Any]],
    *,
    gpu_ids: list[int],
    workers_per_gpu: int,
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, int]]]:
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU IDs must be non-empty and unique")
    if workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be positive")
    worker_count = min(len(jobs), len(gpu_ids) * workers_per_gpu)
    assignments = [
        {"worker_index": index, "gpu_id": gpu_ids[index // workers_per_gpu]}
        for index in range(worker_count)
    ]
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    job_queues = [context.Queue() for _ in range(worker_count)]
    processes = [
        context.Process(
            target=worker_main,
            args=(assignment["gpu_id"], job_queues[assignment["worker_index"]], result_queue, str(output_root)),
            name=f"line-b-parity-gpu-{assignment['gpu_id']}-worker-{assignment['worker_index']}",
        )
        for assignment in assignments
    ]
    for process in processes:
        process.start()
    for job_index, job in enumerate(jobs):
        job_queues[job_index % worker_count].put((job_index, job))
    for job_queue in job_queues:
        job_queue.put(None)

    completed: dict[int, tuple[list[dict[str, Any]], str | None]] = {}
    try:
        while len(completed) < len(jobs):
            try:
                job_index, rows, error = result_queue.get(timeout=5)
            except queue_module.Empty:
                crashed = [p for p in processes if p.exitcode not in (None, 0)]
                if crashed:
                    raise RuntimeError("diagnostic worker crashed: " + ", ".join(f"{p.name}={p.exitcode}" for p in crashed))
                continue
            completed[job_index] = (rows, error)
    finally:
        for process in processes:
            process.join()

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for job_index, job in enumerate(jobs):
        job_rows, error = completed[job_index]
        rows.extend(job_rows)
        if error:
            errors.append(f"{job['task']}: {error}")
    return rows, errors, assignments


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for row in rows:
        task = row["task"]
        role = row["role"]
        bucket = summary.setdefault(task, {}).setdefault(role, {"attempts": 0, "passed": 0, "plan_failed": 0, "check_failed": 0, "other_failed": 0})
        bucket["attempts"] += 1
        if row["passed"]:
            bucket["passed"] += 1
        elif row["error_type"] == "expert_plan_failed":
            bucket["plan_failed"] += 1
        elif row["error_type"] == "expert_check_failed":
            bucket["check_failed"] += 1
        else:
            bucket["other_failed"] += 1
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-seed", action="append", default=[], metavar="TASK=SEED")
    parser.add_argument("--controls-per-task", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--condition", choices=("isolated", "packed", "custom"), required=True)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--workers-per-gpu", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.controls_per_task < 0:
        raise SystemExit("repeats must be positive and controls-per-task non-negative")
    if args.condition == "isolated" and args.workers_per_gpu != 1:
        raise SystemExit("isolated condition requires --workers-per-gpu 1")
    if args.condition == "packed" and args.workers_per_gpu != 3:
        raise SystemExit("packed condition requires --workers-per-gpu 3")

    failure_specs = args.failed_seed or list(DEFAULT_FAILURE_SPECS)
    failures = parse_failure_specs(failure_specs)
    output_dir = args.output_dir or BRACE_DIR / "runs" / f"line_b_collect_parity_{args.condition}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    jobs: list[dict[str, Any]] = []
    manifest_provenance: dict[str, Any] = {}
    for task, seeds in failures.items():
        provenance, cases = build_task_cases(task, seeds, controls_per_task=args.controls_per_task)
        manifest_provenance[task] = provenance
        jobs.append({"task": task, "cases": cases, "repeats": args.repeats})

    started_at = utc_now()
    rows, worker_errors, assignments = run_jobs_parallel(
        jobs,
        gpu_ids=args.gpus,
        workers_per_gpu=args.workers_per_gpu,
        output_root=output_dir,
    )
    report = {
        "schema_version": 1,
        "stage": "line_b_collect_path_parity_diagnostic",
        "diagnostic_only": True,
        "condition": args.condition,
        "started_at": started_at,
        "completed_at": utc_now(),
        "git_commit": git_commit(),
        "source_task_config": "task_config/demo_clean.yml",
        "source_task_config_sha256": file_sha256(REPO_ROOT / "task_config" / "demo_clean.yml"),
        "effective_config_overrides": {"episode_num": 50, "use_seed": True, "need_plan": True, "render_freq": 0, "collect_data": False},
        "gpus": args.gpus,
        "workers_per_gpu": args.workers_per_gpu,
        "worker_assignments": assignments,
        "repeats": args.repeats,
        "controls_per_task": args.controls_per_task,
        "failure_specs": failure_specs,
        "manifest_provenance": manifest_provenance,
        "summary": summarize(rows),
        "worker_errors": worker_errors,
        "rows": rows,
        "notes": [
            "No seed substitution is performed.",
            "Each attempt uses the seed's real frozen-cohort episode index and writes trajectory data only below this run directory.",
            "Run isolated and packed conditions from the same commit before changing manifests or collection retry rules.",
        ],
    }
    write_json_atomic(output_dir / "report.json", report)
    print(json.dumps({"output": str(output_dir / "report.json"), "summary": report["summary"], "worker_errors": worker_errors}, indent=2))
    return 1 if worker_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
