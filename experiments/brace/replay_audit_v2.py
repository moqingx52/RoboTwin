#!/usr/bin/env python3
"""BRACE audit v2: snapshot restore + exact control-trace replay."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import queue as queue_module
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE1_DIR = REPO_ROOT / "experiments" / "phase1"
for import_path in (REPO_ROOT, PHASE1_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from experiments.brace.control_trace import load_brace_trace, robot_state_dict, validate_schema_v2
from experiments.brace.replay_audit import (
    METRIC_TO_THRESHOLD,
    Candidate,
    collect_candidates,
    compare_state,
    file_sha256,
    git_commit,
    read_json,
    repo_path,
    select_candidates,
    thresholds_from_protocol,
    write_json_atomic,
    write_jsonl_atomic,
)


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


def compare_robot_state(actual: dict[str, Any], expected: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    episode = _episode_from_state(expected)
    errors = compare_state(
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
    metric_passed = {metric: value <= thresholds[metric] for metric, value in errors.items()}
    return {"errors": errors, "metric_passed": metric_passed, "passed": all(metric_passed.values())}


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


def audit_snapshot(
    candidate: Candidate,
    snapshot: dict[str, Any],
    trace: dict[str, Any],
    thresholds: dict[str, float],
    *,
    restore_repeat_count: int,
    horizon: int,
    task_config: str,
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

    env = setup_env(candidate, task_config)
    run_args = _make_run_args(candidate, task_config)
    try:
        restored_states = []
        for _ in range(restore_repeat_count):
            env.setup_demo(now_ep_num=0, seed=candidate.env_seed, is_test=True, **run_args)
            env.restore_branch_snapshot(snapshot)
            restored_states.append(robot_state_dict(env))

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

        env.setup_demo(now_ep_num=0, seed=candidate.env_seed, is_test=True, **run_args)
        env.restore_branch_snapshot(snapshot)
        for step in trace["control_steps"][start_index:end_index]:
            env.replay_control_step(step)

        expected_state = trace["control_steps"][end_index]["robot_state"]
        actual_state = robot_state_dict(env)
        comparison = compare_robot_state(actual_state, expected_state, thresholds)
        rows.append(
            {
                **base_row,
                "check_type": "control_trace_replay",
                "horizon_physics_steps": int(actual_horizon),
                **comparison,
            }
        )
    finally:
        try:
            env.close_env()
        except Exception:
            pass
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
            )
        )
    return rows


def audit_worker(gpu: int, jobs: Any, results: Any) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    while True:
        job = jobs.get()
        if job is None:
            return
        (
            job_index,
            candidate,
            snapshots,
            thresholds,
            restore_repeat_count,
            horizon,
            task_config,
        ) = job
        try:
            trace = load_brace_trace(candidate.path)
            rows = audit_candidate_snapshots(
                candidate,
                snapshots,
                trace,
                thresholds,
                restore_repeat_count=restore_repeat_count,
                horizon=horizon,
                task_config=task_config,
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
    work: list[tuple[Candidate, list[dict[str, Any]]]],
    thresholds: dict[str, float],
    gpus: list[int],
    *,
    restore_repeat_count: int,
    horizon: int,
    task_config: str,
) -> tuple[list[dict[str, Any]], list[tuple[Candidate, str]]]:
    if not work:
        return [], []
    if not gpus:
        raise ValueError("at least one GPU is required")
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"GPU IDs must be unique, got {gpus}")

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    worker_count = min(len(gpus), len(work))
    job_queues = [context.Queue() for _ in range(worker_count)]
    workers = [
        context.Process(
            target=audit_worker,
            args=(gpus[index], job_queues[index], result_queue),
            name=f"replay-audit-v2-gpu-{gpus[index]}",
        )
        for index in range(worker_count)
    ]
    for worker in workers:
        worker.start()

    try:
        for job_index, (candidate, snapshots) in enumerate(work):
            job_queues[job_index % worker_count].put(
                (
                    job_index,
                    candidate,
                    snapshots,
                    thresholds,
                    restore_repeat_count,
                    horizon,
                    task_config,
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
                    f"{worker.name} exitcode={worker.exitcode}"
                    for worker in workers
                    if worker.exitcode not in (None, 0)
                ]
                if crashed:
                    raise RuntimeError(
                        "replay audit v2 worker exited before returning its result: "
                        + ", ".join(crashed)
                    )
                continue
            completed[job_index] = (rows, error)
            print(
                f"Replay audit v2 trajectory {len(completed)}/{len(work)} completed",
                flush=True,
            )
    except BaseException:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        raise
    finally:
        for worker in workers:
            worker.join()
        for job_queue in job_queues:
            job_queue.close()
        result_queue.close()

    checks: list[dict[str, Any]] = []
    errors: list[tuple[Candidate, str]] = []
    for job_index, (candidate, _) in enumerate(work):
        rows, error = completed[job_index]
        checks.extend(rows)
        if error is not None:
            errors.append((candidate, error))
    return checks, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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

    checks: list[dict[str, Any]] = []
    preflight_errors: list[str] = []
    task_summaries: dict[str, Any] = {}
    selected_by_task: dict[str, list[Candidate]] = {}

    for task_index, task in enumerate(protocol["tasks"]):
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
        work: list[tuple[Candidate, list[dict[str, Any]]]] = []
        for task in protocol["tasks"]:
            for candidate in selected_by_task[task]:
                schema_errors = validate_schema_v2(candidate.path)
                if schema_errors:
                    message = f"{task}: invalid trace seed={candidate.env_seed} rollout={candidate.rollout_id}"
                    preflight_errors.append(message)
                    task_summaries[task]["preflight_errors"].append(message)
                    continue
                trace = load_brace_trace(candidate.path)
                snapshots = trace["branch_snapshots"][:snapshot_count]
                if len(snapshots) < snapshot_count:
                    message = (
                        f"{task}: seed={candidate.env_seed} rollout={candidate.rollout_id} "
                        f"has {len(snapshots)} snapshots, need {snapshot_count}"
                    )
                    preflight_errors.append(message)
                    task_summaries[task]["preflight_errors"].append(message)
                    continue
                work.append((candidate, snapshots))

        if not preflight_errors and work:
            parallel_checks, audit_errors = audit_candidates_parallel(
                work,
                thresholds,
                args.gpus,
                restore_repeat_count=restore_repeat_count,
                horizon=horizon,
                task_config=task_config,
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
        task_summary.update(
            {
                "total_checks": len(task_checks),
                "passed_checks": task_passed,
                "pass_rate": task_passed / len(task_checks) if task_checks else 0.0,
            }
        )

    expected_checks = (
        sum(int(task["requested_trajectories"]) * int(task["snapshots_per_trajectory"]) * 2 for task in task_summaries.values())
    )
    passed_checks = sum(bool(row["passed"]) for row in checks)
    total_checks = len(checks)
    pass_rate = passed_checks / total_checks if total_checks else 0.0
    complete = total_checks == expected_checks and not preflight_errors
    passed = complete and pass_rate >= float(protocol["replay_gate"]["minimum_pass_rate"])
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
        "passed": passed,
        "complete": complete,
        "pass_rate": pass_rate,
        "minimum_pass_rate": float(protocol["replay_gate"]["minimum_pass_rate"]),
        "passed_checks": passed_checks,
        "total_checks": total_checks,
        "expected_checks": expected_checks,
        "failed_checks": total_checks - passed_checks,
        "protocol_path": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "git_commit": git_commit(),
        "thresholds": thresholds,
        "metric_errors": metric_errors,
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
    print(
        f"Replay audit v2 passed={summary['passed']} complete={summary['complete']} "
        f"passed_checks={summary['passed_checks']}/{summary['total_checks']} "
        f"expected_checks={summary['expected_checks']} summary={output_dir / 'summary.json'}"
    )
    if preflight_errors:
        print(preflight_errors[0], file=sys.stderr)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
