#!/usr/bin/env python3
"""Collect BRACE Stage-2 branch verification pilots."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import queue as queue_module
import random
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE1_DIR = REPO_ROOT / "experiments" / "phase1"
for import_path in (REPO_ROOT, PHASE1_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from experiments.brace.control_trace import load_brace_trace, policy_chunk_actions
from experiments.brace.replay_audit import (
    Candidate,
    collect_candidates,
    file_sha256,
    git_commit,
    read_json,
    repo_path,
    write_json_atomic,
    write_jsonl_atomic,
)
from experiments.brace.replay_audit_v2 import worker_gpu_assignments, job_worker_index, resolve_worker_count


@dataclass(frozen=True)
class BranchPoint:
    task: str
    env_seed: int
    physics_step: int
    snapshot_id: int
    point_type: str
    success_rollout_id: int
    failure_rollout_id: int


@dataclass(frozen=True)
class BranchJob:
    point: BranchPoint
    candidate_chunk_index: int
    control_chunk_indices: tuple[int, ...]
    continuation_seeds: tuple[int, ...]


def _steps_by_physics_step(trace: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(step["physics_step"]): step for step in trace["control_steps"]}


def _joint_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_joints = np.asarray(left["robot_state"]["joints"], dtype=np.float64)
    right_joints = np.asarray(right["robot_state"]["joints"], dtype=np.float64)
    return float(np.linalg.norm(left_joints - right_joints))


def _snapshot_for_physics_step(trace: dict[str, Any], physics_step: int) -> dict[str, Any] | None:
    for snapshot in trace["branch_snapshots"]:
        if int(snapshot["physics_step"]) == int(physics_step):
            return snapshot
    if trace["branch_snapshots"]:
        return min(
            trace["branch_snapshots"],
            key=lambda item: abs(int(item["physics_step"]) - int(physics_step)),
        )
    return None


def _policy_chunk_index_for_physics_step(trace: dict[str, Any], physics_step: int) -> int:
    for chunk in trace["policy_chunks"]:
        start = int(chunk["physics_step"])
        actions = policy_chunk_actions(chunk["action"])
        end = start + len(actions)
        if start <= physics_step < end:
            return int(chunk["chunk_index"])
    if trace["policy_chunks"]:
        return int(trace["policy_chunks"][-1]["chunk_index"])
    return 0


def select_branch_points(
    task: str,
    env_seed: int,
    success: Candidate,
    failure: Candidate,
    *,
    max_points: int,
    rng: random.Random,
) -> list[BranchPoint]:
    success_trace = load_brace_trace(success.path)
    failure_trace = load_brace_trace(failure.path)
    success_steps = _steps_by_physics_step(success_trace)
    failure_steps = _steps_by_physics_step(failure_trace)
    shared_steps = sorted(set(success_steps) & set(failure_steps))
    if len(shared_steps) < 3:
        return []

    divergences = [_joint_distance(success_steps[step], failure_steps[step]) for step in shared_steps]
    points: list[BranchPoint] = []

    first_persistent = None
    streak = 0
    threshold = 0.02
    for step, divergence in zip(shared_steps, divergences):
        if divergence >= threshold:
            streak += 1
            if streak >= 3 and first_persistent is None:
                first_persistent = step
        else:
            streak = 0
    if first_persistent is not None:
        snapshot = _snapshot_for_physics_step(success_trace, first_persistent)
        if snapshot is not None:
            points.append(
                BranchPoint(
                    task=task,
                    env_seed=env_seed,
                    physics_step=int(first_persistent),
                    snapshot_id=int(snapshot["snapshot_id"]),
                    point_type="first_persistent_divergence",
                    success_rollout_id=success.rollout_id,
                    failure_rollout_id=failure.rollout_id,
                )
            )

    interior = shared_steps[1:-1]
    if interior:
        peak_step = interior[int(np.argmax([divergences[shared_steps.index(step)] for step in interior]))]
        snapshot = _snapshot_for_physics_step(success_trace, peak_step)
        if snapshot is not None and all(point.physics_step != peak_step for point in points):
            points.append(
                BranchPoint(
                    task=task,
                    env_seed=env_seed,
                    physics_step=int(peak_step),
                    snapshot_id=int(snapshot["snapshot_id"]),
                    point_type="local_divergence_peak",
                    success_rollout_id=success.rollout_id,
                    failure_rollout_id=failure.rollout_id,
                )
            )

    negative_pool = [step for step in interior if all(point.physics_step != step for point in points)]
    rng.shuffle(negative_pool)
    for step in negative_pool[: max(0, max_points - len(points))]:
        snapshot = _snapshot_for_physics_step(success_trace, step)
        if snapshot is None:
            continue
        points.append(
            BranchPoint(
                task=task,
                env_seed=env_seed,
                physics_step=int(step),
                snapshot_id=int(snapshot["snapshot_id"]),
                point_type="random_negative_control",
                success_rollout_id=success.rollout_id,
                failure_rollout_id=failure.rollout_id,
            )
        )
        if len(points) >= max_points:
            break
    return points[:max_points]


def build_branch_jobs(
    task: str,
    rollout_dir: Path,
    env_seeds: list[int],
    *,
    max_points: int,
    control_k: int,
    continuation_m: int,
    selection_seed: int,
) -> list[BranchJob]:
    candidates, _errors = collect_candidates(task, rollout_dir)
    by_seed: dict[int, dict[bool, list[Candidate]]] = {}
    for candidate in candidates:
        by_seed.setdefault(candidate.env_seed, {}).setdefault(candidate.success, []).append(candidate)

    jobs: list[BranchJob] = []
    for seed_index, env_seed in enumerate(env_seeds):
        seed_candidates = by_seed.get(env_seed, {})
        successes = seed_candidates.get(True, [])
        failures = seed_candidates.get(False, [])
        if not successes or not failures:
            continue
        success = sorted(successes, key=lambda item: item.rollout_id)[0]
        failure = sorted(failures, key=lambda item: item.rollout_id)[0]
        rng = random.Random(selection_seed + seed_index)
        points = select_branch_points(
            task,
            env_seed,
            success,
            failure,
            max_points=max_points,
            rng=rng,
        )
        success_trace = load_brace_trace(success.path)
        chunk_count = len(success_trace["policy_chunks"])
        if chunk_count == 0:
            continue
        continuation_seeds = tuple(range(continuation_m))
        for point in points:
            candidate_chunk_index = _policy_chunk_index_for_physics_step(success_trace, point.physics_step)
            control_pool = [index for index in range(chunk_count) if index != candidate_chunk_index]
            rng.shuffle(control_pool)
            control_indices = tuple(control_pool[:control_k])
            if len(control_indices) < control_k:
                continue
            jobs.append(
                BranchJob(
                    point=point,
                    candidate_chunk_index=candidate_chunk_index,
                    control_chunk_indices=control_indices,
                    continuation_seeds=continuation_seeds,
                )
            )
    return jobs


def _chunk_actions(trace: dict[str, Any], chunk_index: int) -> np.ndarray:
    for chunk in trace["policy_chunks"]:
        if int(chunk["chunk_index"]) == int(chunk_index):
            return policy_chunk_actions(chunk["action"])
    raise ValueError(f"chunk_index {chunk_index} not found")


def _run_branch_episode(
    env,
    *,
    snapshot: dict[str, Any],
    chunk_actions: np.ndarray,
    continuation_seed: int,
    model,
    encode_obs,
) -> dict[str, Any]:
    import torch

    if hasattr(model, "set_generator"):
        generator = torch.Generator(device="cuda:0")
        generator.manual_seed(int(continuation_seed))
        model.set_generator(generator)

    model.reset_obs()
    env.restore_branch_snapshot(snapshot)
    observation = env.get_obs()
    obs = encode_obs(observation)
    transitions = 0
    for action in chunk_actions:
        env.take_action(np.asarray(action, dtype=np.float64))
        transitions += 1
        observation = env.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)
        if env.eval_success:
            break

    while env.take_action_cnt < env.step_lim and not env.eval_success:
        actions = model.get_action(obs)
        for action in actions:
            env.take_action(action)
            transitions += 1
            observation = env.get_obs()
            obs = encode_obs(observation)
            model.update_obs(obs)
            if env.eval_success:
                break
        if env.eval_success:
            break

    return {
        "success": bool(env.eval_success),
        "transitions": int(transitions),
        "physics_steps": int(env.physics_step),
    }


def run_branch_job(
    job: BranchJob,
    rollout_dir: Path,
    *,
    task_config: str,
    model_args: dict[str, Any],
) -> list[dict[str, Any]]:
    from common import load_task_args, make_task_env
    from policy.DP.deploy_policy import encode_obs, get_model

    task = job.point.task
    task_dir = rollout_dir / task
    candidates, _ = collect_candidates(task, rollout_dir)
    success = next(
        item
        for item in candidates
        if item.env_seed == job.point.env_seed and item.rollout_id == job.point.success_rollout_id
    )
    failure = next(
        item
        for item in candidates
        if item.env_seed == job.point.env_seed and item.rollout_id == job.point.failure_rollout_id
    )
    success_trace = load_brace_trace(success.path)
    failure_trace = load_brace_trace(failure.path)
    snapshot = next(
        item for item in success_trace["branch_snapshots"] if int(item["snapshot_id"]) == int(job.point.snapshot_id)
    )

    env_args = load_task_args(task, task_config)
    run_args = dict(env_args)
    run_args.update({"need_plan": False, "save_data": False, "eval_mode": True, "render_freq": 0})
    model = get_model(model_args)

    rows: list[dict[str, Any]] = []
    candidate_actions = _chunk_actions(success_trace, job.candidate_chunk_index)
    for continuation_seed in job.continuation_seeds:
        env = make_task_env(task)
        try:
            env.setup_demo(now_ep_num=0, seed=job.point.env_seed, is_test=True, **run_args)
            result = _run_branch_episode(
                env,
                snapshot=snapshot,
                chunk_actions=candidate_actions,
                continuation_seed=continuation_seed,
                model=model,
                encode_obs=encode_obs,
            )
            rows.append(
                {
                    **job.point.__dict__,
                    "branch_role": "candidate",
                    "chunk_index": job.candidate_chunk_index,
                    "continuation_seed": continuation_seed,
                    **result,
                }
            )
        finally:
            try:
                env.close_env()
            except Exception:
                pass

    for control_chunk_index in job.control_chunk_indices:
        control_actions = _chunk_actions(failure_trace, control_chunk_index)
        for continuation_seed in job.continuation_seeds:
            env = make_task_env(task)
            try:
                env.setup_demo(now_ep_num=0, seed=job.point.env_seed, is_test=True, **run_args)
                result = _run_branch_episode(
                    env,
                    snapshot=snapshot,
                    chunk_actions=control_actions,
                    continuation_seed=continuation_seed,
                    model=model,
                    encode_obs=encode_obs,
                )
                rows.append(
                    {
                        **job.point.__dict__,
                        "branch_role": "control",
                        "chunk_index": control_chunk_index,
                        "continuation_seed": continuation_seed,
                        **result,
                    }
                )
            finally:
                try:
                    env.close_env()
                except Exception:
                    pass
    return rows


def bootstrap_lcb(successes: list[bool], *, alpha: float, samples: int, rng: random.Random) -> float:
    if not successes:
        return 0.0
    values = np.asarray(successes, dtype=np.float64)
    n = len(values)
    boot = []
    for _ in range(samples):
        draw = values[[rng.randrange(n) for _ in range(n)]]
        boot.append(float(draw.mean()))
    return float(np.quantile(boot, alpha))


def summarize_branch_rows(rows: list[dict[str, Any]], *, alpha: float, delta: float) -> dict[str, Any]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["env_seed"]), int(row["physics_step"]))
        grouped.setdefault(key, []).append(row)

    point_summaries = []
    accepted = 0
    for (_env_seed, _physics_step), group in grouped.items():
        candidate_success = [bool(row["success"]) for row in group if row["branch_role"] == "candidate"]
        control_success = [bool(row["success"]) for row in group if row["branch_role"] == "control"]
        advantage = (sum(candidate_success) / len(candidate_success) if candidate_success else 0.0) - (
            sum(control_success) / len(control_success) if control_success else 0.0
        )
        rng = random.Random(_env_seed + _physics_step)
        lcb = bootstrap_lcb(candidate_success, alpha=alpha, samples=500, rng=rng) - (
            sum(control_success) / len(control_success) if control_success else 0.0
        )
        accepted_flag = lcb > delta
        accepted += int(accepted_flag)
        point_summaries.append(
            {
                "env_seed": _env_seed,
                "physics_step": _physics_step,
                "point_type": group[0]["point_type"],
                "candidate_success_rate": sum(candidate_success) / len(candidate_success),
                "control_success_rate": sum(control_success) / len(control_success),
                "advantage": advantage,
                "lcb_advantage": lcb,
                "accepted": accepted_flag,
            }
        )

    recovery_seeds = {
        row["env_seed"]
        for row in point_summaries
        if row["point_type"] != "random_negative_control" and row["candidate_success_rate"] > 0
    }
    mixed_outcome_seeds = {row["env_seed"] for row in rows}
    recovery_fraction = (
        len(recovery_seeds) / len(mixed_outcome_seeds) if mixed_outcome_seeds else 0.0
    )
    matched_lift = np.mean(
        [row["candidate_success_rate"] - row["control_success_rate"] for row in point_summaries]
    ) if point_summaries else 0.0
    passed = matched_lift >= 0.20 or recovery_fraction >= 0.20
    return {
        "passed": bool(passed),
        "accepted_points": accepted,
        "total_points": len(point_summaries),
        "matched_success_lift": float(matched_lift),
        "recovery_seed_fraction": float(recovery_fraction),
        "points": point_summaries,
    }


def branch_worker(gpu_id: int, jobs: Any, results: Any, rollout_dir: Path, task_config: str, model_args: dict[str, Any]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    while True:
        job = jobs.get()
        if job is None:
            return
        job_index, branch_job = job
        try:
            rows = run_branch_job(branch_job, rollout_dir, task_config=task_config, model_args=model_args)
            results.put((job_index, rows, None))
        except BaseException as exc:
            results.put((job_index, [], f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))


def run_branch_jobs_parallel(
    jobs: list[BranchJob],
    rollout_dir: Path,
    *,
    workers: int,
    gpu_ids: list[int],
    workers_per_gpu: int,
    task_config: str,
    model_args: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    if not jobs:
        return [], []
    worker_count = resolve_worker_count(
        work_size=len(jobs),
        workers=workers,
        workers_per_gpu=workers_per_gpu,
        gpu_ids=gpu_ids,
    )
    assignments = worker_gpu_assignments(gpu_ids, workers_per_gpu)[:worker_count]
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    job_queues = [context.Queue() for _ in range(worker_count)]
    processes = [
        context.Process(
            target=branch_worker,
            args=(gpu_id, job_queues[worker_index], result_queue, rollout_dir, task_config, model_args),
            name=f"collect-branches-gpu-{gpu_id}-worker-{worker_index}",
        )
        for worker_index, gpu_id in assignments
    ]
    for process in processes:
        process.start()
    try:
        for job_index, branch_job in enumerate(jobs):
            job_queues[job_worker_index(job_index, worker_count)].put((job_index, branch_job))
        for job_queue in job_queues:
            job_queue.put(None)

        completed: dict[int, tuple[list[dict[str, Any]], str | None]] = {}
        while len(completed) < len(jobs):
            job_index, rows, error = result_queue.get()
            completed[job_index] = (rows, error)
            print(f"  branch jobs {len(completed)}/{len(jobs)}", flush=True)
    finally:
        for process in processes:
            process.join()

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for job_index in range(len(jobs)):
        job_rows, error = completed[job_index]
        rows.extend(job_rows)
        if error is not None:
            errors.append(error)
    return rows, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--seeds-file", type=Path, default=None)
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol_path = repo_path(args.protocol)
    rollout_dir = repo_path(args.rollout_dir)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = read_json(protocol_path)
    tasks = list(args.tasks) if args.tasks else list(protocol["tasks"])
    task_config = protocol.get("task_config", "demo_brace_trace")
    max_points = int(protocol.get("candidate_points_per_pair", 4))
    control_k = int(protocol.get("control_chunks_k", 3))
    continuation_m = int(protocol.get("continuation_seeds_m", 3))
    alpha = float(protocol["acceptance"]["one_sided_alpha"])
    delta = float(protocol["acceptance"]["minimum_advantage_delta"])

    all_rows: list[dict[str, Any]] = []
    task_summaries: dict[str, Any] = {}
    errors: list[str] = []
    budgets: dict[str, Any] = {}

    for task in tasks:
        seeds_file = args.seeds_file or (REPO_ROOT / "experiments" / "brace" / "seeds" / f"{task}_pilot_seeds.json")
        if not seeds_file.is_file():
            errors.append(f"{task}: missing pilot seeds file {seeds_file}")
            continue
        seeds_payload = read_json(seeds_file)
        env_seeds = [int(seed) for seed in seeds_payload["seeds"]]
        jobs = build_branch_jobs(
            task,
            rollout_dir,
            env_seeds,
            max_points=max_points,
            control_k=control_k,
            continuation_m=continuation_m,
            selection_seed=int(protocol["replay_audit"].get("selection_seed", 0)),
        )
        print(f"Collecting branches for {task}: {len(jobs)} jobs across {len(env_seeds)} seeds", flush=True)
        workers = args.workers if args.workers is not None else len(args.gpus) * args.workers_per_gpu
        model_args = {
            "task_name": task,
            "task_config": task_config,
            "ckpt_setting": "demo_clean",
            "expert_data_num": 200,
            "seed": 0,
            "checkpoint_num": 600,
            "left_arm_dim": 6,
            "right_arm_dim": 6,
        }
        rows, job_errors = run_branch_jobs_parallel(
            jobs,
            rollout_dir,
            workers=workers,
            gpu_ids=args.gpus,
            workers_per_gpu=args.workers_per_gpu,
            task_config=task_config,
            model_args=model_args,
        )
        errors.extend(job_errors)
        all_rows.extend(rows)
        summary = summarize_branch_rows(rows, alpha=alpha, delta=delta)
        task_summaries[task] = summary
        budgets[task] = {
            "environment_transitions": int(sum(int(row.get("transitions", 0)) for row in rows)),
            "accepted_chunks": int(summary["accepted_points"]),
            "optimizer_examples": 0,
        }
        budget_path = REPO_ROOT / "experiments" / "brace" / "budgets" / f"{task}_pilot.json"
        budget_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(budget_path, budgets[task])

    passed = bool(task_summaries) and all(summary["passed"] for summary in task_summaries.values()) and not errors
    summary = {
        "schema_version": 2,
        "protocol_revision": protocol.get("protocol_revision", "2.2"),
        "passed": passed,
        "complete": not errors,
        "tasks": task_summaries,
        "errors": errors,
        "protocol_path": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "git_commit": git_commit(),
        "artifacts": {"checks": "checks.jsonl", "budgets": "../budgets"},
    }
    write_jsonl_atomic(output_dir / "checks.jsonl", all_rows)
    write_json_atomic(output_dir / "summary.json", summary)
    print(f"Branch collection passed={passed} summary={output_dir / 'summary.json'}")
    if errors:
        print(errors[0], file=sys.stderr)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
