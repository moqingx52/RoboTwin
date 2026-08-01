#!/usr/bin/env python3
"""BRACE audit v2: snapshot restore + exact control-trace replay."""

from __future__ import annotations

import argparse
import fnmatch
import json
import multiprocessing
import os
import queue as queue_module
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE1_DIR = REPO_ROOT / "experiments" / "phase1"
for import_path in (REPO_ROOT, PHASE1_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from experiments.brace.control_trace import load_brace_trace, robot_state_dict
from experiments.brace.replay_audit import (
    METRIC_TO_THRESHOLD,
    Candidate,
    collect_candidates,
    compare_state_detailed,
    file_sha256,
    git_commit,
    read_json,
    repo_path,
    select_candidates,
    thresholds_from_protocol,
    write_json_atomic,
    write_jsonl_atomic,
)


def worker_gpu_assignments(gpu_ids: list[int], workers_per_gpu: int) -> list[tuple[int, int]]:
    """Map each worker index to a fixed GPU id."""
    if workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be positive")
    if not gpu_ids:
        raise ValueError("at least one GPU id is required")
    return [
        (worker_index, gpu_ids[worker_index // workers_per_gpu])
        for worker_index in range(len(gpu_ids) * workers_per_gpu)
    ]


def job_worker_index(job_index: int, worker_count: int) -> int:
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    return job_index % worker_count


def resolve_worker_count(
    *,
    work_size: int,
    workers: int,
    workers_per_gpu: int,
    gpu_ids: list[int],
) -> int:
    configured = len(gpu_ids) * workers_per_gpu
    if workers != configured:
        print(
            f"Warning: --workers={workers} does not match "
            f"{len(gpu_ids)} gpus x {workers_per_gpu} workers/gpu={configured}; "
            f"using {configured}",
            flush=True,
        )
    worker_count = min(configured, work_size)
    return max(worker_count, 1)


def _buffer_index_for_physics_step(control_steps: list[dict[str, Any]], physics_step: int) -> int:
    for index, step in enumerate(control_steps):
        if int(step["physics_step"]) == int(physics_step):
            return index
    raise ValueError(f"physics_step {physics_step} not found in control trace")


def _state_tuple(state: dict[str, Any]) -> tuple[np.ndarray, ...]:
    parts = [
        np.asarray(state["joints"], dtype=np.float64),
        np.asarray(state["left_endpose"], dtype=np.float64),
        np.asarray(state["right_endpose"], dtype=np.float64),
    ]
    for name in sorted(state["dynamic_actors"]):
        actor = state["dynamic_actors"][name]
        parts.append(np.asarray(actor["pose"], dtype=np.float64))
        parts.append(np.asarray(actor["linear_velocity"], dtype=np.float64))
        parts.append(np.asarray(actor["angular_velocity"], dtype=np.float64))
    return tuple(parts)


def _states_match(left: dict[str, Any], right: dict[str, Any], atol: float = 1e-9) -> bool:
    return all(np.allclose(a, b, atol=atol, rtol=0.0) for a, b in zip(_state_tuple(left), _state_tuple(right)))


def _episode_from_state(state: dict[str, Any]):
    from experiments.brace.replay_audit import Episode

    objects = {name: np.asarray(actor["pose"], dtype=np.float64) for name, actor in state["dynamic_actors"].items()}
    return Episode(
        joints=np.asarray(state["joints"], dtype=np.float64)[None, :],
        end_effectors={
            "left_endpose": np.asarray(state["left_endpose"], dtype=np.float64)[None, :],
            "right_endpose": np.asarray(state["right_endpose"], dtype=np.float64)[None, :],
        },
        objects={name: value[None, :] for name, value in objects.items()},
    )


def actor_metrics_for_task(protocol: dict[str, Any], task: str) -> dict[str, list[str]]:
    actor_metrics = protocol.get("actor_metrics", {})
    task_metrics = actor_metrics.get(task, {})
    if not isinstance(task_metrics, dict):
        raise ValueError(f"actor_metrics.{task} must be an object")
    return {str(pattern): list(metrics) for pattern, metrics in task_metrics.items()}


def resolve_actor_metric_dims(actor_name: str, task_metrics: dict[str, list[str]]) -> list[str]:
    for pattern, metrics in task_metrics.items():
        if fnmatch.fnmatch(actor_name, pattern):
            return list(metrics)
    return ["translation", "rotation"]


def apply_actor_metrics(
    errors: dict[str, Any],
    thresholds: dict[str, float],
    task_metrics: dict[str, list[str]],
) -> dict[str, Any]:
    actor_errors = errors.get("actor_errors", {})
    actor_metric_passed: dict[str, dict[str, bool]] = {}
    rotation_values: list[float] = []
    translation_values: list[float] = []

    for actor_name, metrics in sorted(actor_errors.items()):
        dims = resolve_actor_metric_dims(actor_name, task_metrics)
        actor_passed: dict[str, bool] = {}
        if "translation" in dims:
            value = float(metrics["translation_error"])
            translation_values.append(value)
            actor_passed["translation"] = value <= thresholds["object_translation_error"]
        if "rotation" in dims:
            value = float(metrics["rotation_error"])
            rotation_values.append(value)
            actor_passed["rotation"] = value <= thresholds["object_rotation_error"]
        actor_metric_passed[actor_name] = actor_passed

    gated_errors = {
        "joint_max_error": errors["joint_max_error"],
        "end_effector_translation_error": errors["end_effector_translation_error"],
        "end_effector_rotation_error": errors["end_effector_rotation_error"],
        "object_translation_error": max(translation_values) if translation_values else 0.0,
        "object_rotation_error": max(rotation_values) if rotation_values else 0.0,
        "actor_errors": actor_errors,
    }
    metric_passed = {
        metric: gated_errors[metric] <= thresholds[metric]
        for metric in METRIC_TO_THRESHOLD
    }
    worst_actor = None
    worst_metric = None
    worst_value = -1.0
    for actor_name, metrics in sorted(actor_errors.items()):
        dims = resolve_actor_metric_dims(actor_name, task_metrics)
        for metric_name in dims:
            key = f"{metric_name}_error"
            value = float(metrics[key])
            if not actor_metric_passed[actor_name].get(metric_name, True) and value >= worst_value:
                worst_actor = actor_name
                worst_metric = metric_name
                worst_value = value

    passed = all(metric_passed.values())
    return {
        "errors": gated_errors,
        "metric_passed": metric_passed,
        "actor_metric_passed": actor_metric_passed,
        "worst_actor": worst_actor,
        "worst_metric": worst_metric,
        "passed": passed,
    }


def compare_robot_state(
    actual: dict[str, Any],
    expected: dict[str, Any],
    thresholds: dict[str, float],
    *,
    task_metrics: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    episode = _episode_from_state(expected)
    errors = compare_state_detailed(
        (
            np.asarray(actual["joints"], dtype=np.float64),
            {
                "left_endpose": np.asarray(actual["left_endpose"], dtype=np.float64),
                "right_endpose": np.asarray(actual["right_endpose"], dtype=np.float64),
            },
            {name: np.asarray(actor["pose"], dtype=np.float64) for name, actor in actual["dynamic_actors"].items()},
        ),
        episode,
        0,
    )
    if task_metrics:
        return apply_actor_metrics(errors, thresholds, task_metrics)
    scalar_errors = {metric: errors[metric] for metric in METRIC_TO_THRESHOLD}
    metric_passed = {metric: value <= thresholds[metric] for metric, value in scalar_errors.items()}
    worst_actor = None
    worst_metric = None
    worst_value = -1.0
    for actor_name, metrics in sorted(errors.get("actor_errors", {}).items()):
        for metric_name, threshold_key in (
            ("translation", "object_translation_error"),
            ("rotation", "object_rotation_error"),
        ):
            value = float(metrics[f"{metric_name}_error"])
            if value > thresholds[threshold_key] and value >= worst_value:
                worst_actor = actor_name
                worst_metric = metric_name
                worst_value = value
    return {
        "errors": errors,
        "metric_passed": metric_passed,
        "actor_metric_passed": {},
        "worst_actor": worst_actor,
        "worst_metric": worst_metric,
        "passed": all(metric_passed.values()),
    }


def _make_run_args(candidate: Candidate, task_config: str) -> dict[str, Any]:
    from common import load_task_args

    env_args = load_task_args(candidate.task, task_config)
    run_args = dict(env_args)
    run_args.update({"need_plan": False, "save_data": False, "eval_mode": True, "render_freq": 0})
    return run_args


def setup_env(candidate: Candidate, task_config: str):
    from common import make_task_env

    env = make_task_env(candidate.task)
    env.setup_demo(
        now_ep_num=0,
        seed=candidate.env_seed,
        is_test=True,
        **_make_run_args(candidate, task_config),
    )
    return env


def _run_with_env(candidate: Candidate, task_config: str, fn: Callable[[Any], Any]):
    env = setup_env(candidate, task_config)
    try:
        return fn(env)
    finally:
        try:
            env.close_env()
        except Exception:
            pass


def audit_snapshot(
    candidate: Candidate,
    snapshot: dict[str, Any],
    trace: dict[str, Any],
    thresholds: dict[str, float],
    *,
    restore_repeat_count: int,
    horizon: int,
    task_config: str,
    task_metrics: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_row = {
        "task": candidate.task,
        "env_seed": candidate.env_seed,
        "rollout_id": candidate.rollout_id,
        "success": candidate.success,
        "hdf5_path": str(candidate.path),
        "snapshot_id": int(snapshot["snapshot_id"]),
        "physics_step": int(snapshot["physics_step"]),
    }

    restored_states = []
    for _ in range(restore_repeat_count):
        def _capture_restore_state(env):
            env.restore_branch_snapshot(snapshot)
            return robot_state_dict(env)

        restored_states.append(_run_with_env(candidate, task_config, _capture_restore_state))

    restore_passed = all(_states_match(restored_states[0], state) for state in restored_states[1:])
    rows.append(
        {
            **base_row,
            "check_type": "restore_determinism",
            "horizon_physics_steps": 0,
            "errors": {},
            "metric_passed": {"restore_determinism": restore_passed},
            "passed": restore_passed,
        }
    )

    start_index = _buffer_index_for_physics_step(trace["control_steps"], snapshot["physics_step"])
    end_index = min(start_index + horizon, len(trace["control_steps"]) - 1)
    actual_horizon = end_index - start_index
    if actual_horizon <= 0:
        raise ValueError(f"snapshot {snapshot['snapshot_id']} has no replay horizon")

    def _replay_control_trace(env):
        env.restore_branch_snapshot(snapshot)
        for step in trace["control_steps"][start_index:end_index]:
            env.replay_control_step(step)
        expected_state = trace["control_steps"][end_index]["robot_state"]
        actual_state = robot_state_dict(env)
        return compare_robot_state(actual_state, expected_state, thresholds, task_metrics=task_metrics)

    comparison = _run_with_env(candidate, task_config, _replay_control_trace)
    rows.append(
        {
            **base_row,
            "check_type": "control_trace_replay",
            "horizon_physics_steps": int(actual_horizon),
            **comparison,
        }
    )
    return rows


def audit_candidate_snapshots(
    candidate: Candidate,
    snapshots: list[dict[str, Any]],
    trace: dict[str, Any],
    thresholds: dict[str, float],
    *,
    restore_repeat_count: int,
    horizon: int,
    task_config: str,
    task_metrics: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        rows.extend(
            audit_snapshot(
                candidate,
                snapshot,
                trace,
                thresholds,
                restore_repeat_count=restore_repeat_count,
                horizon=horizon,
                task_config=task_config,
                task_metrics=task_metrics,
            )
        )
    return rows


def run_audit_job(
    candidate: Candidate,
    snapshot_count: int,
    thresholds: dict[str, float],
    *,
    restore_repeat_count: int,
    horizon: int,
    task_config: str,
    task_metrics: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    trace = load_brace_trace(candidate.path)
    snapshots = trace["branch_snapshots"][:snapshot_count]
    if len(snapshots) < snapshot_count:
        raise ValueError(
            f"seed={candidate.env_seed} rollout={candidate.rollout_id} "
            f"has {len(snapshots)} snapshots, need {snapshot_count}"
        )
    return audit_candidate_snapshots(
        candidate,
        snapshots,
        trace,
        thresholds,
        restore_repeat_count=restore_repeat_count,
        horizon=horizon,
        task_config=task_config,
        task_metrics=task_metrics,
    )


def audit_worker(gpu_id: int, jobs: Any, results: Any) -> None:
    """Audit trajectories in a simulator process pinned to one GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    while True:
        job = jobs.get()
        if job is None:
            return
        (
            job_index,
            candidate,
            snapshot_count,
            thresholds,
            restore_repeat_count,
            horizon,
            task_config,
            task_metrics,
        ) = job
        try:
            rows = run_audit_job(
                candidate,
                snapshot_count,
                thresholds,
                restore_repeat_count=restore_repeat_count,
                horizon=horizon,
                task_config=task_config,
                task_metrics=task_metrics,
            )
            results.put((job_index, rows, None))
        except BaseException as exc:
            results.put(
                (
                    job_index,
                    [],
                    f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                )
            )


def audit_candidates_parallel(
    work: list[Candidate],
    thresholds: dict[str, float],
    workers: int,
    gpu_ids: list[int],
    *,
    workers_per_gpu: int,
    snapshot_count: int,
    restore_repeat_count: int,
    horizon: int,
    task_config: str,
    task_metrics_by_task: dict[str, dict[str, list[str]]],
) -> tuple[list[dict[str, Any]], list[tuple[Candidate, str]]]:
    if not work:
        return [], []
    if not gpu_ids:
        raise ValueError("at least one GPU id is required for simulator import (curobo)")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"GPU IDs must be unique, got {gpu_ids}")

    worker_count = resolve_worker_count(
        work_size=len(work),
        workers=workers,
        workers_per_gpu=workers_per_gpu,
        gpu_ids=gpu_ids,
    )
    assignments = worker_gpu_assignments(gpu_ids, workers_per_gpu)[:worker_count]
    print(
        f"Starting replay audit v2 on {len(work)} trajectories with "
        f"workers={worker_count} workers_per_gpu={workers_per_gpu} gpus={gpu_ids} "
        f"(one HDF5 load per trajectory)",
        flush=True,
    )

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    job_queues = [context.Queue() for _ in range(worker_count)]
    processes = [
        context.Process(
            target=audit_worker,
            args=(gpu_id, job_queues[worker_index], result_queue),
            name=f"replay-audit-v2-gpu-{gpu_id}-worker-{worker_index}",
        )
        for worker_index, gpu_id in assignments
    ]
    for process in processes:
        process.start()

    try:
        for job_index, candidate in enumerate(work):
            worker_index = job_worker_index(job_index, worker_count)
            job_queues[worker_index].put(
                (
                    job_index,
                    candidate,
                    snapshot_count,
                    thresholds,
                    restore_repeat_count,
                    horizon,
                    task_config,
                    task_metrics_by_task.get(candidate.task),
                )
            )
        for job_queue in job_queues:
            job_queue.put(None)

        completed: dict[int, tuple[list[dict[str, Any]], str | None]] = {}
        while len(completed) < len(work):
            try:
                job_index, rows, error = result_queue.get(timeout=5)
            except queue_module.Empty:
                crashed = [
                    f"{process.name} exitcode={process.exitcode}"
                    for process in processes
                    if process.exitcode not in (None, 0)
                ]
                if crashed:
                    raise RuntimeError(
                        "replay audit v2 worker exited before returning its result: "
                        + ", ".join(crashed)
                    )
                continue
            completed[job_index] = (rows, error)
            print(
                f"  audited {len(completed)}/{len(work)} trajectories",
                flush=True,
            )
    except BaseException:
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise
    finally:
        for process in processes:
            process.join()
        for job_queue in job_queues:
            job_queue.close()
        result_queue.close()

    checks: list[dict[str, Any]] = []
    errors: list[tuple[Candidate, str]] = []
    for job_index, candidate in enumerate(work):
        rows, error = completed[job_index]
        checks.extend(rows)
        if error is not None:
            errors.append((candidate, error))
    return checks, errors


def summarize_check_type(checks: list[dict[str, Any]], check_type: str) -> dict[str, Any]:
    subset = [row for row in checks if row.get("check_type") == check_type]
    passed_checks = sum(bool(row["passed"]) for row in subset)
    total_checks = len(subset)
    return {
        "passed_checks": passed_checks,
        "total_checks": total_checks,
        "failed_checks": total_checks - passed_checks,
        "pass_rate": passed_checks / total_checks if total_checks else 0.0,
    }


def replay_gate_requirements(protocol: dict[str, Any]) -> dict[str, float]:
    gate = protocol["replay_gate"]
    replay_required = float(
        gate.get(
            "control_trace_replay_minimum_pass_rate",
            gate.get("minimum_pass_rate", 0.95),
        )
    )
    return {
        "restore_determinism_pass_rate": float(gate.get("restore_determinism_pass_rate", 1.0)),
        "control_trace_replay_minimum_pass_rate": replay_required,
        "per_task_control_trace_replay_minimum_pass_rate": float(
            gate.get("per_task_control_trace_replay_minimum_pass_rate", replay_required)
        ),
    }


def evaluate_task_replay_gate(
    checks: list[dict[str, Any]],
    task: str,
    requirements: dict[str, float],
) -> dict[str, Any]:
    restore = summarize_check_type(
        [row for row in checks if row["task"] == task],
        "restore_determinism",
    )
    replay = summarize_check_type(
        [row for row in checks if row["task"] == task],
        "control_trace_replay",
    )
    restore_passed = (
        restore["total_checks"] > 0
        and restore["pass_rate"] >= requirements["restore_determinism_pass_rate"]
        and restore["failed_checks"] == 0
    )
    replay_passed = (
        replay["total_checks"] > 0
        and replay["pass_rate"] >= requirements["per_task_control_trace_replay_minimum_pass_rate"]
    )
    return {
        "restore_determinism": {**restore, "passed": restore_passed},
        "control_trace_replay": {**replay, "passed": replay_passed},
        "replay_gate_passed": restore_passed and replay_passed,
    }


def evaluate_replay_gate(
    checks: list[dict[str, Any]],
    tasks: list[str],
    protocol: dict[str, Any],
    *,
    complete: bool,
    preflight_errors: list[str],
) -> dict[str, Any]:
    requirements = replay_gate_requirements(protocol)
    restore_stats = summarize_check_type(checks, "restore_determinism")
    replay_stats = summarize_check_type(checks, "control_trace_replay")

    restore_passed = (
        restore_stats["total_checks"] > 0
        and restore_stats["pass_rate"] >= requirements["restore_determinism_pass_rate"]
        and restore_stats["failed_checks"] == 0
    )
    replay_passed = (
        replay_stats["total_checks"] > 0
        and replay_stats["pass_rate"] >= requirements["control_trace_replay_minimum_pass_rate"]
    )

    per_task: dict[str, Any] = {}
    for task in tasks:
        per_task[task] = evaluate_task_replay_gate(checks, task, requirements)

    per_task_passed = all(task_stats["replay_gate_passed"] for task_stats in per_task.values()) if per_task else False
    passed = complete and not preflight_errors and restore_passed and replay_passed and per_task_passed

    passed_checks = sum(bool(row["passed"]) for row in checks)
    total_checks = len(checks)
    return {
        "passed": passed,
        "complete": complete,
        "preflight_errors": preflight_errors,
        "restore_determinism": {
            **restore_stats,
            "passed": restore_passed,
            "required_pass_rate": requirements["restore_determinism_pass_rate"],
        },
        "control_trace_replay": {
            **replay_stats,
            "passed": replay_passed,
            "required_pass_rate": requirements["control_trace_replay_minimum_pass_rate"],
        },
        "per_task_control_trace_replay_minimum_pass_rate": requirements[
            "per_task_control_trace_replay_minimum_pass_rate"
        ],
        "per_task": per_task,
        "mixed_pass_rate": passed_checks / total_checks if total_checks else 0.0,
        "passed_checks": passed_checks,
        "total_checks": total_checks,
        "failed_checks": total_checks - passed_checks,
    }


def summarize_per_actor_failures(checks: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in checks:
        if row.get("check_type") != "control_trace_replay" or row.get("passed"):
            continue
        task = row["task"]
        actor_errors = row.get("errors", {}).get("actor_errors", {})
        actor_metric_passed = row.get("actor_metric_passed", {})
        for actor_name, metrics in actor_errors.items():
            task_counts = counts.setdefault(task, {})
            passed = actor_metric_passed.get(actor_name, {})
            if passed.get("translation") is False:
                key = f"{actor_name}:translation"
                task_counts[key] = task_counts.get(key, 0) + 1
            if passed.get("rotation") is False:
                key = f"{actor_name}:rotation"
                task_counts[key] = task_counts.get(key, 0) + 1
            if not actor_metric_passed:
                if float(metrics["rotation_error"]) > 0.05:
                    key = f"{actor_name}:rotation"
                    task_counts[key] = task_counts.get(key, 0) + 1
                if float(metrics["translation_error"]) > 0.005:
                    key = f"{actor_name}:translation"
                    task_counts[key] = task_counts.get(key, 0) + 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional task subset; defaults to all tasks in the protocol.",
    )
    parser.add_argument(
        "--horizon-curve",
        type=str,
        default=None,
        help="Comma-separated replay horizons for diagnostic sweeps (e.g. 1,5,10,20,50).",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=3,
        help="Fixed simulator workers per GPU (matches traced collection default).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Total worker count; defaults to len(gpus) * workers_per_gpu.",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=list(range(8)),
        help="GPU ids pinned at worker creation (required for curobo import).",
    )
    return parser.parse_args()


def _run_audit_for_horizons(
    *,
    protocol: dict[str, Any],
    rollout_dir: Path,
    output_dir: Path,
    tasks: list[str],
    thresholds: dict[str, float],
    workers: int,
    gpu_ids: list[int],
    workers_per_gpu: int,
    horizons: list[int],
    task_metrics_by_task: dict[str, dict[str, list[str]]],
) -> int:
    sampling = protocol["replay_audit"]
    trajectory_count = int(sampling["trajectories_per_task"])
    snapshot_count = int(sampling["snapshots_per_trajectory"])
    selection_seed = int(sampling["selection_seed"])
    require_both = bool(sampling["require_success_and_failure"])
    restore_repeat_count = int(sampling["restore_repeat_count"])
    task_config = protocol.get("task_config", "demo_brace_trace")

    curve_summary: dict[str, Any] = {"horizons": horizons, "tasks": {}}
    exit_code = 0
    for horizon in horizons:
        checks: list[dict[str, Any]] = []
        preflight_errors: list[str] = []
        selected_by_task: dict[str, list[Candidate]] = {}
        for task_index, task in enumerate(tasks):
            candidates, errors = collect_candidates(task, rollout_dir)
            selected, selection_errors = select_candidates(
                candidates,
                trajectory_count,
                selection_seed + task_index,
                require_both,
            )
            preflight_errors.extend(f"{task}: {error}" for error in errors + selection_errors)
            selected_by_task[task] = selected

        if not preflight_errors:
            work = [candidate for task in tasks for candidate in selected_by_task[task]]
            if work:
                parallel_checks, audit_errors = audit_candidates_parallel(
                    work,
                    thresholds,
                    workers,
                    gpu_ids,
                    workers_per_gpu=workers_per_gpu,
                    snapshot_count=snapshot_count,
                    restore_repeat_count=restore_repeat_count,
                    horizon=horizon,
                    task_config=task_config,
                    task_metrics_by_task=task_metrics_by_task,
                )
                checks.extend(parallel_checks)
                for candidate, error in audit_errors:
                    preflight_errors.append(
                        f"{candidate.task}: audit failed seed={candidate.env_seed} "
                        f"rollout={candidate.rollout_id}: {error}"
                    )

        gate_summary = evaluate_replay_gate(
            checks,
            tasks,
            protocol,
            complete=not preflight_errors,
            preflight_errors=preflight_errors,
        )
        curve_summary["tasks"][str(horizon)] = {
            "passed": gate_summary["passed"],
            "control_trace_replay": gate_summary["control_trace_replay"],
            "per_task": gate_summary["per_task"],
            "per_actor_failure_counts": summarize_per_actor_failures(checks),
        }
        horizon_dir = output_dir / f"horizon_{horizon}"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl_atomic(horizon_dir / "checks.jsonl", checks)
        write_jsonl_atomic(horizon_dir / "failures.jsonl", [row for row in checks if not row["passed"]])
        if not gate_summary["passed"]:
            exit_code = 1

    write_json_atomic(output_dir / "horizon_curve_summary.json", curve_summary)
    print(f"Horizon curve summary written to {output_dir / 'horizon_curve_summary.json'}")
    return exit_code


def main() -> int:
    args = parse_args()
    if args.workers_per_gpu < 1:
        raise SystemExit("--workers-per-gpu must be >= 1")
    if not args.gpus:
        raise SystemExit("at least one --gpus id is required")
    if len(set(args.gpus)) != len(args.gpus):
        raise SystemExit("--gpus ids must be unique")
    workers = args.workers if args.workers is not None else len(args.gpus) * args.workers_per_gpu
    if workers < 1:
        raise SystemExit("--workers must be >= 1")

    protocol_path = repo_path(args.protocol)
    rollout_dir = repo_path(args.rollout_dir)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        protocol = read_json(protocol_path)
        if protocol.get("schema_version") != 2:
            raise ValueError("protocol.v2 requires schema_version=2")
        if protocol["replay_audit"].get("mode") != "snapshot_control_trace":
            raise ValueError("unsupported replay_audit.mode")
        thresholds = thresholds_from_protocol(protocol)
        sampling = protocol["replay_audit"]
        trajectory_count = int(sampling["trajectories_per_task"])
        snapshot_count = int(sampling["snapshots_per_trajectory"])
        selection_seed = int(sampling["selection_seed"])
        require_both = bool(sampling["require_success_and_failure"])
        restore_repeat_count = int(sampling["restore_repeat_count"])
        horizon = int(sampling["continuation_horizon_physics_steps"])
        task_config = protocol.get("task_config", "demo_brace_trace")
        tasks = list(args.tasks) if args.tasks else list(protocol["tasks"])
        for task in tasks:
            if task not in protocol["tasks"]:
                raise ValueError(f"unknown task {task!r} in protocol tasks={protocol['tasks']}")
        task_metrics_by_task = {
            task: actor_metrics_for_task(protocol, task)
            for task in tasks
            if protocol.get("actor_metrics", {}).get(task)
        }
    except Exception as exc:
        summary = {
            "schema_version": 2,
            "passed": False,
            "complete": False,
            "preflight_errors": [f"invalid protocol: {type(exc).__name__}: {exc}"],
        }
        write_json_atomic(output_dir / "summary.json", summary)
        print(summary["preflight_errors"][0], file=sys.stderr)
        return 2

    if args.horizon_curve:
        horizons = [int(value.strip()) for value in args.horizon_curve.split(",") if value.strip()]
        return _run_audit_for_horizons(
            protocol=protocol,
            rollout_dir=rollout_dir,
            output_dir=output_dir,
            tasks=tasks,
            thresholds=thresholds,
            workers=workers,
            gpu_ids=args.gpus,
            workers_per_gpu=args.workers_per_gpu,
            horizons=horizons,
            task_metrics_by_task=task_metrics_by_task,
        )

    checks: list[dict[str, Any]] = []
    preflight_errors: list[str] = []
    task_summaries: dict[str, Any] = {}
    selected_by_task: dict[str, list[Candidate]] = {}

    for task_index, task in enumerate(tasks):
        candidates, errors = collect_candidates(task, rollout_dir)
        selected, selection_errors = select_candidates(
            candidates,
            trajectory_count,
            selection_seed + task_index,
            require_both,
        )
        all_errors = errors + selection_errors
        preflight_errors.extend(f"{task}: {error}" for error in all_errors)
        selected_by_task[task] = selected
        task_summaries[task] = {
            "available_trajectories": len(candidates),
            "available_successes": sum(item.success for item in candidates),
            "available_failures": sum(not item.success for item in candidates),
            "requested_trajectories": trajectory_count,
            "selected_trajectories": len(selected),
            "selected_successes": sum(item.success for item in selected),
            "selected_failures": sum(not item.success for item in selected),
            "snapshots_per_trajectory": snapshot_count,
            "preflight_errors": all_errors,
        }

    if not preflight_errors:
        work = [candidate for task in tasks for candidate in selected_by_task[task]]
        if work:
            parallel_checks, audit_errors = audit_candidates_parallel(
                work,
                thresholds,
                workers,
                args.gpus,
                workers_per_gpu=args.workers_per_gpu,
                snapshot_count=snapshot_count,
                restore_repeat_count=restore_repeat_count,
                horizon=horizon,
                task_config=task_config,
                task_metrics_by_task=task_metrics_by_task,
            )
            checks.extend(parallel_checks)
            for candidate, error in audit_errors:
                message = (
                    f"{candidate.task}: audit failed seed={candidate.env_seed} "
                    f"rollout={candidate.rollout_id}: {error}"
                )
                preflight_errors.append(message)
                task_summaries[candidate.task]["preflight_errors"].append(message)

    for task, task_summary in task_summaries.items():
        task_checks = [row for row in checks if row["task"] == task]
        task_passed = sum(bool(row["passed"]) for row in task_checks)
        task_gate = evaluate_task_replay_gate(checks, task, replay_gate_requirements(protocol))
        task_summary.update(
            {
                "total_checks": len(task_checks),
                "passed_checks": task_passed,
                "pass_rate": task_passed / len(task_checks) if task_checks else 0.0,
                "restore_determinism": task_gate["restore_determinism"],
                "control_trace_replay": task_gate["control_trace_replay"],
                "replay_gate_passed": task_gate["replay_gate_passed"],
            }
        )

    expected_checks = (
        sum(int(task["requested_trajectories"]) * int(task["snapshots_per_trajectory"]) * 2 for task in task_summaries.values())
    )
    expected_restore_checks = expected_checks // 2
    expected_replay_checks = expected_checks // 2
    complete = len(checks) == expected_checks and not preflight_errors
    gate_summary = evaluate_replay_gate(
        checks,
        tasks,
        protocol,
        complete=complete,
        preflight_errors=preflight_errors,
    )
    passed = gate_summary["passed"]
    metric_errors = {
        metric: {
            "maximum": max(values),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
        }
        for metric in METRIC_TO_THRESHOLD
        if (values := [float(row["errors"][metric]) for row in checks if row.get("errors") and metric in row["errors"]])
    }
    summary = {
        "schema_version": 2,
        "protocol_revision": protocol.get("protocol_revision", "2.0"),
        "passed": passed,
        "complete": complete,
        "pass_rate": gate_summary["mixed_pass_rate"],
        "minimum_pass_rate": float(protocol["replay_gate"].get("minimum_pass_rate", 0.95)),
        "passed_checks": gate_summary["passed_checks"],
        "total_checks": gate_summary["total_checks"],
        "expected_checks": expected_checks,
        "expected_restore_checks": expected_restore_checks,
        "expected_replay_checks": expected_replay_checks,
        "failed_checks": gate_summary["failed_checks"],
        "restore_determinism": gate_summary["restore_determinism"],
        "control_trace_replay": gate_summary["control_trace_replay"],
        "per_task_control_trace_replay_minimum_pass_rate": gate_summary[
            "per_task_control_trace_replay_minimum_pass_rate"
        ],
        "protocol_path": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "git_commit": git_commit(),
        "thresholds": thresholds,
        "metric_errors": metric_errors,
        "per_actor_failure_counts": summarize_per_actor_failures(checks),
        "tasks": task_summaries,
        "preflight_errors": preflight_errors,
        "artifacts": {
            "checks": "checks.jsonl",
            "failures": "failures.jsonl",
            "diagnostics": "diagnostics.jsonl",
        },
    }
    write_jsonl_atomic(output_dir / "checks.jsonl", checks)
    write_jsonl_atomic(output_dir / "failures.jsonl", [row for row in checks if not row["passed"]])
    write_jsonl_atomic(output_dir / "diagnostics.jsonl", [{"error": error} for error in preflight_errors])
    write_json_atomic(output_dir / "summary.json", summary)
    try:
        from experiments.brace.stage_records import emit_stage_record

        emit_stage_record(
            "audit_v2",
            summary=summary,
            summary_path=output_dir / "summary.json",
            tasks=list(args.tasks),
        )
    except Exception as exc:  # noqa: BLE001 — record emission must not fail the audit
        print(f"Warning: failed to emit stage record: {exc}", file=sys.stderr)
    print(
        f"Replay audit v2 passed={summary['passed']} complete={summary['complete']} "
        f"restore={summary['restore_determinism']['passed_checks']}/"
        f"{summary['restore_determinism']['total_checks']} "
        f"replay={summary['control_trace_replay']['passed_checks']}/"
        f"{summary['control_trace_replay']['total_checks']} "
        f"(mixed={summary['passed_checks']}/{summary['total_checks']}) "
        f"summary={output_dir / 'summary.json'}"
    )
    if preflight_errors:
        print(preflight_errors[0], file=sys.stderr)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
