#!/usr/bin/env python3
"""Collect BRACE Stage-2 branch verification pilots."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
import queue as queue_module
import random
import sys
import time
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

from experiments.brace.control_trace import (
    BranchContext,
    build_branch_context,
    load_brace_trace,
    policy_chunk_actions,
    restore_model_obs_history,
    trace_has_chunk_index,
)
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
    snapshot_id: int
    snapshot_physics_step: int
    physics_step: int
    branch_chunk_index: int
    point_type: str
    success_rollout_id: int


@dataclass(frozen=True)
class BranchJob:
    point: BranchPoint
    candidate_chunk_index: int
    control_failure_rollout_ids: tuple[int, ...]
    continuation_seeds: tuple[int, ...]


def _steps_by_physics_step(trace: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(step["physics_step"]): step for step in trace["control_steps"]}


def _joint_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_joints = np.asarray(left["robot_state"]["joints"], dtype=np.float64)
    right_joints = np.asarray(right["robot_state"]["joints"], dtype=np.float64)
    return float(np.linalg.norm(left_joints - right_joints))


def select_branch_points(
    task: str,
    env_seed: int,
    success: Candidate,
    failures: list[Candidate],
    *,
    max_points: int,
    rng: random.Random,
    success_trace: dict[str, Any] | None = None,
    failure_traces: list[dict[str, Any]] | None = None,
) -> list[BranchPoint]:
    """Select branch points from consensus divergence over all failure rollouts.

    For every chunk-boundary snapshot of the success trace, the consensus
    divergence d_k is the mean joint-space distance between the success state
    and every failure rollout's state at the same physics step. The divergence
    growth onset is k* = argmax_k (d_{k+1} - d_k); the causal branch point is
    placed one boundary earlier (k* - 1), before the trajectories separate, and
    a matched random control is drawn from boundaries outside the growth window
    {k*-1, k*, k*+1}.
    """
    if success_trace is None:
        success_trace = load_brace_trace(success.path)
    failure_list = list(failures)
    if failure_traces is None:
        failure_traces = [load_brace_trace(failure.path) for failure in failure_list]

    success_steps = _steps_by_physics_step(success_trace)
    failure_steps_list = [_steps_by_physics_step(trace) for trace in failure_traces]

    scored: list[tuple[float, dict[str, Any], BranchContext]] = []
    seen_snapshot_ids: set[int] = set()
    for snapshot in success_trace["branch_snapshots"]:
        snapshot_id = int(snapshot["snapshot_id"])
        if snapshot_id in seen_snapshot_ids:
            continue
        snapshot_physics_step = int(snapshot["physics_step"])
        if snapshot_physics_step not in success_steps:
            continue
        distances = [
            _joint_distance(success_steps[snapshot_physics_step], failure_steps[snapshot_physics_step])
            for failure_steps in failure_steps_list
            if snapshot_physics_step in failure_steps
        ]
        if not distances:
            continue
        try:
            context = build_branch_context(success_trace, snapshot)
        except ValueError:
            continue
        seen_snapshot_ids.add(snapshot_id)
        scored.append((float(np.mean(distances)), snapshot, context))

    if not scored:
        return []

    chronological = sorted(scored, key=lambda item: int(item[1]["physics_step"]))
    divergences = [item[0] for item in chronological]
    if len(chronological) >= 2:
        growth_index = max(
            range(len(divergences) - 1),
            key=lambda index: divergences[index + 1] - divergences[index],
        )
        causal_index = max(0, growth_index - 1)
    else:
        growth_index = 0
        causal_index = 0

    selected: list[tuple[str, tuple[float, dict[str, Any], BranchContext]]] = [
        ("pre_divergence_causal", chronological[causal_index])
    ]
    growth_window = {causal_index, growth_index, growth_index + 1}
    control_pool = [
        chronological[index] for index in range(len(chronological)) if index not in growth_window
    ]
    if not control_pool:
        control_pool = [
            chronological[index] for index in range(len(chronological)) if index != causal_index
        ]
    if control_pool:
        selected.append(("matched_random_control", rng.choice(control_pool)))

    points: list[BranchPoint] = []
    for point_type, (_divergence, snapshot, context) in selected[:max_points]:
        points.append(
            BranchPoint(
                task=task,
                env_seed=env_seed,
                snapshot_id=int(snapshot["snapshot_id"]),
                snapshot_physics_step=int(snapshot["physics_step"]),
                physics_step=int(context.boundary_physics_step),
                branch_chunk_index=int(context.branch_chunk_index),
                point_type=point_type,
                success_rollout_id=success.rollout_id,
            )
        )
    return points


def _select_control_failure_rollout_ids(
    failures: list[Candidate],
    *,
    branch_chunk_index: int,
    control_k: int,
    rng: random.Random,
) -> tuple[int, ...]:
    eligible: list[Candidate] = []
    for failure in failures:
        failure_trace = load_brace_trace(failure.path)
        if trace_has_chunk_index(failure_trace, branch_chunk_index):
            eligible.append(failure)
    rng.shuffle(eligible)
    return tuple(failure.rollout_id for failure in eligible[:control_k])


def build_branch_jobs(
    task: str,
    rollout_dir: Path,
    env_seeds: list[int],
    *,
    max_points: int,
    control_k: int,
    continuation_m: int,
    selection_seed: int,
    prepare_workers: int = 1,
) -> list[BranchJob]:
    started_at = time.monotonic()
    print(f"[branch prepare] task={task} phase=manifest status=starting dir={rollout_dir / task}", flush=True)
    candidates, _errors = collect_candidates(
        task,
        rollout_dir,
        progress_callback=lambda rows: print(
            f"[branch prepare] task={task} phase=manifest status=scanning rows={rows}",
            flush=True,
        ),
        progress_every=25,
    )
    success_count = sum(candidate.success for candidate in candidates)
    failure_count = len(candidates) - success_count
    print(
        f"[branch prepare] task={task} phase=manifest status=done "
        f"candidates={len(candidates)} successes={success_count} failures={failure_count} "
        f"elapsed={time.monotonic() - started_at:.1f}s",
        flush=True,
    )
    for error in _errors:
        print(f"[branch prepare] task={task} phase=manifest warning={error}", flush=True)

    by_seed: dict[int, dict[bool, list[Candidate]]] = {}
    for candidate in candidates:
        by_seed.setdefault(candidate.env_seed, {}).setdefault(candidate.success, []).append(candidate)

    seed_inputs: list[tuple[int, str, int, Candidate, tuple[Candidate, ...], int, int, int, int]] = []
    for seed_index, env_seed in enumerate(env_seeds):
        seed_candidates = by_seed.get(env_seed, {})
        successes = seed_candidates.get(True, [])
        failures = seed_candidates.get(False, [])
        if not successes or not failures:
            print(
                f"[branch prepare] task={task} phase=pairing seed={env_seed} status=skipped "
                f"successes={len(successes)} failures={len(failures)}",
                flush=True,
            )
            continue
        success = sorted(successes, key=lambda item: item.rollout_id)[0]
        failure_tuple = tuple(sorted(failures, key=lambda item: item.rollout_id))
        seed_inputs.append(
            (
                seed_index,
                task,
                env_seed,
                success,
                failure_tuple,
                max_points,
                control_k,
                continuation_m,
                selection_seed,
            )
        )

    if not seed_inputs:
        print(f"[branch prepare] task={task} phase=trace-analysis status=done seeds=0 jobs=0", flush=True)
        return []

    worker_count = min(max(prepare_workers, 1), len(seed_inputs))
    print(
        f"[branch prepare] task={task} phase=trace-analysis status=starting "
        f"seeds={len(seed_inputs)} workers={worker_count}",
        flush=True,
    )
    jobs_by_seed_index: dict[int, list[BranchJob]] = {}
    if worker_count == 1:
        results = map(_build_seed_branch_jobs, seed_inputs)
        for completed_count, (seed_index, env_seed, seed_jobs) in enumerate(results, start=1):
            jobs_by_seed_index[seed_index] = seed_jobs
            print(
                f"[branch prepare] task={task} phase=trace-analysis progress={completed_count}/{len(seed_inputs)} "
                f"seed={env_seed} jobs={len(seed_jobs)} elapsed={time.monotonic() - started_at:.1f}s",
                flush=True,
            )
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as pool:
            futures = [pool.submit(_build_seed_branch_jobs, item) for item in seed_inputs]
            for completed_count, future in enumerate(as_completed(futures), start=1):
                seed_index, env_seed, seed_jobs = future.result()
                jobs_by_seed_index[seed_index] = seed_jobs
                print(
                    f"[branch prepare] task={task} phase=trace-analysis progress={completed_count}/{len(seed_inputs)} "
                    f"seed={env_seed} jobs={len(seed_jobs)} elapsed={time.monotonic() - started_at:.1f}s",
                    flush=True,
                )

    jobs = [job for seed_index in sorted(jobs_by_seed_index) for job in jobs_by_seed_index[seed_index]]
    print(
        f"[branch prepare] task={task} phase=trace-analysis status=done "
        f"seeds={len(seed_inputs)} jobs={len(jobs)} elapsed={time.monotonic() - started_at:.1f}s",
        flush=True,
    )
    return jobs


def _build_seed_branch_jobs(
    item: tuple[int, str, int, Candidate, tuple[Candidate, ...], int, int, int, int],
) -> tuple[int, int, list[BranchJob]]:
    seed_index, task, env_seed, success, failures, max_points, control_k, continuation_m, selection_seed = item
    rng = random.Random(selection_seed + seed_index)
    success_trace = load_brace_trace(success.path)
    failure_list = list(failures)
    failure_traces = [load_brace_trace(failure.path) for failure in failure_list]
    points = select_branch_points(
        task,
        env_seed,
        success,
        failure_list,
        max_points=max_points,
        rng=rng,
        success_trace=success_trace,
        failure_traces=failure_traces,
    )
    jobs: list[BranchJob] = []
    if not points:
        return seed_index, env_seed, jobs

    continuation_seeds = tuple(range(continuation_m))
    for point in points:
        control_failure_rollout_ids = _select_control_failure_rollout_ids(
            list(failures),
            branch_chunk_index=point.branch_chunk_index,
            control_k=control_k,
            rng=rng,
        )
        if len(control_failure_rollout_ids) < control_k:
            continue
        jobs.append(
            BranchJob(
                point=point,
                candidate_chunk_index=point.branch_chunk_index,
                control_failure_rollout_ids=control_failure_rollout_ids,
                continuation_seeds=continuation_seeds,
            )
        )
    return seed_index, env_seed, jobs


def _chunk_actions(trace: dict[str, Any], chunk_index: int) -> np.ndarray:
    for chunk in trace["policy_chunks"]:
        if int(chunk["chunk_index"]) == int(chunk_index):
            return policy_chunk_actions(chunk["action"])
    raise ValueError(f"chunk_index {chunk_index} not found")


def _run_branch_episode(
    env,
    *,
    snapshot: dict[str, Any],
    replay_steps: list[dict[str, Any]],
    runtime_state: dict[str, Any],
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

    # Restore the observation frames the policy had consumed at this chunk
    # boundary during collection (falls back to a bare reset for legacy traces
    # without stored history). The boundary frame is re-appended to match the
    # collection-time deque state right after get_action produced this chunk.
    boundary_frame = restore_model_obs_history(model, snapshot.get("obs_history"))
    env.restore_branch_snapshot(snapshot)
    for step in replay_steps:
        env.replay_control_step(step)
    env.apply_branch_runtime_state(runtime_state)
    if boundary_frame is not None:
        model.update_obs(boundary_frame)

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


def _branch_row_fields(point: BranchPoint, branch_context: BranchContext) -> dict[str, Any]:
    return {
        **point.__dict__,
        "boundary_physics_step": int(branch_context.boundary_physics_step),
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
    candidates, _ = collect_candidates(task, rollout_dir)
    success = next(
        item
        for item in candidates
        if item.env_seed == job.point.env_seed and item.rollout_id == job.point.success_rollout_id
    )
    success_trace = load_brace_trace(success.path)
    snapshot = next(
        item for item in success_trace["branch_snapshots"] if int(item["snapshot_id"]) == int(job.point.snapshot_id)
    )
    branch_context = build_branch_context(success_trace, snapshot)

    env_args = load_task_args(task, task_config)
    run_args = dict(env_args)
    run_args.update({"need_plan": False, "save_data": False, "eval_mode": True, "render_freq": 0})
    model = get_model(model_args)

    rows: list[dict[str, Any]] = []
    candidate_actions = _chunk_actions(success_trace, job.candidate_chunk_index)
    base_fields = _branch_row_fields(job.point, branch_context)
    for continuation_seed in job.continuation_seeds:
        env = make_task_env(task)
        try:
            env.setup_demo(now_ep_num=0, seed=job.point.env_seed, is_test=True, **run_args)
            result = _run_branch_episode(
                env,
                snapshot=snapshot,
                replay_steps=branch_context.replay_steps,
                runtime_state=branch_context.runtime_state,
                chunk_actions=candidate_actions,
                continuation_seed=continuation_seed,
                model=model,
                encode_obs=encode_obs,
            )
            rows.append(
                {
                    **base_fields,
                    "branch_role": "candidate",
                    "chunk_index": job.candidate_chunk_index,
                    "failure_rollout_id": None,
                    "continuation_seed": continuation_seed,
                    **result,
                }
            )
        finally:
            try:
                env.close_env()
            except Exception:
                pass

    for failure_rollout_id in job.control_failure_rollout_ids:
        failure = next(
            item
            for item in candidates
            if item.env_seed == job.point.env_seed and item.rollout_id == failure_rollout_id
        )
        failure_trace = load_brace_trace(failure.path)
        control_actions = _chunk_actions(failure_trace, job.candidate_chunk_index)
        for continuation_seed in job.continuation_seeds:
            env = make_task_env(task)
            try:
                env.setup_demo(now_ep_num=0, seed=job.point.env_seed, is_test=True, **run_args)
                result = _run_branch_episode(
                    env,
                    snapshot=snapshot,
                    replay_steps=branch_context.replay_steps,
                    runtime_state=branch_context.runtime_state,
                    chunk_actions=control_actions,
                    continuation_seed=continuation_seed,
                    model=model,
                    encode_obs=encode_obs,
                )
                rows.append(
                    {
                        **base_fields,
                        "branch_role": "control",
                        "chunk_index": job.candidate_chunk_index,
                        "failure_rollout_id": int(failure_rollout_id),
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


def paired_cluster_lcb(
    group: list[dict[str, Any]], *, alpha: float, samples: int, rng: random.Random
) -> tuple[float, list[float]]:
    """LCB for candidate-control lift, paired and clustered by continuation seed.

    Candidate outcomes are compared with the mean of controls sharing the same
    continuation RNG seed. A bounded finite-cluster correction prevents the
    degenerate 3/3 -> LCB=1 result of the old candidate-only bootstrap.
    """
    candidates: dict[int, list[float]] = {}
    controls: dict[int, list[float]] = {}
    for row in group:
        continuation = int(row.get("continuation_seed", -1))
        target = candidates if row.get("branch_role") == "candidate" else controls
        target.setdefault(continuation, []).append(1.0 if row.get("success") else 0.0)
    paired = []
    for continuation in sorted(set(candidates) & set(controls)):
        candidate_mean = float(np.mean(candidates[continuation]))
        control_mean = float(np.mean(controls[continuation]))
        paired.append(candidate_mean - control_mean)
    if not paired:
        return -1.0, []
    n = len(paired)
    boot = [
        float(np.mean([paired[rng.randrange(n)] for _ in range(n)]))
        for _ in range(samples)
    ]
    percentile_lcb = float(np.quantile(boot, alpha))
    bounded_radius = float(np.sqrt(2.0 * np.log(1.0 / alpha) / n))
    return max(-1.0, percentile_lcb - bounded_radius), paired


def evaluate_harness_sanity(rows: list[dict[str, Any]], *, min_rate: float) -> tuple[bool, str | None]:
    candidate_rows = [row for row in rows if row.get("branch_role") == "candidate"]
    if not candidate_rows:
        return False, "no_candidate_rollouts"
    rate = sum(bool(row["success"]) for row in candidate_rows) / len(candidate_rows)
    if rate < min_rate:
        return False, f"candidate_baseline_rate_{rate:.3f}_below_{min_rate}"
    return True, None


def summarize_branch_rows(rows: list[dict[str, Any]], *, alpha: float, delta: float) -> dict[str, Any]:
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["env_seed"]), int(row["snapshot_id"]), int(row["physics_step"]))
        grouped.setdefault(key, []).append(row)

    point_summaries = []
    accepted = 0
    for (_env_seed, _snapshot_id, _physics_step), group in grouped.items():
        candidate_success = [bool(row["success"]) for row in group if row["branch_role"] == "candidate"]
        control_success = [bool(row["success"]) for row in group if row["branch_role"] == "control"]
        advantage = (sum(candidate_success) / len(candidate_success) if candidate_success else 0.0) - (
            sum(control_success) / len(control_success) if control_success else 0.0
        )
        rng = random.Random(_env_seed + _snapshot_id + _physics_step)
        lcb, paired_differences = paired_cluster_lcb(
            group, alpha=alpha, samples=2000, rng=rng
        )
        accepted_flag = lcb > delta
        accepted += int(accepted_flag)
        point_summaries.append(
            {
                "env_seed": _env_seed,
                "snapshot_id": _snapshot_id,
                "physics_step": _physics_step,
                "snapshot_physics_step": int(group[0]["snapshot_physics_step"]),
                "branch_chunk_index": int(group[0]["branch_chunk_index"]),
                "point_type": group[0]["point_type"],
                "candidate_success_rate": sum(candidate_success) / len(candidate_success),
                "control_success_rate": sum(control_success) / len(control_success),
                "advantage": advantage,
                "lcb_advantage": lcb,
                "lcb_method": "paired_continuation_cluster_bootstrap_with_bounded_small_n_correction",
                "paired_cluster_count": len(paired_differences),
                "paired_cluster_differences": paired_differences,
                "accepted": accepted_flag,
            }
        )

    recovery_seeds = {
        row["env_seed"]
        for row in point_summaries
        if row["point_type"] not in ("random_negative_control", "matched_random_control")
        and row["candidate_success_rate"] > 0
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
        results.put(("started", job_index, os.getpid(), gpu_id))
        try:
            rows = run_branch_job(branch_job, rollout_dir, task_config=task_config, model_args=model_args)
            results.put(("done", job_index, rows, None))
        except BaseException as exc:
            results.put(("done", job_index, [], f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))


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
    started_at = time.monotonic()
    print(
        f"[branch execute] status=launching jobs={len(jobs)} workers={worker_count} "
        f"workers_per_gpu={workers_per_gpu} gpus={gpu_ids}",
        flush=True,
    )
    for process in processes:
        process.start()
    print(
        f"[branch execute] status=workers-started pids={[process.pid for process in processes]}",
        flush=True,
    )
    try:
        for job_index, branch_job in enumerate(jobs):
            job_queues[job_worker_index(job_index, worker_count)].put((job_index, branch_job))
        for job_queue in job_queues:
            job_queue.put(None)

        completed: dict[int, tuple[list[dict[str, Any]], str | None]] = {}
        started_jobs: set[int] = set()
        while len(completed) < len(jobs):
            try:
                message = result_queue.get(timeout=15)
            except queue_module.Empty:
                crashed = [
                    f"{process.name}(pid={process.pid}, exitcode={process.exitcode})"
                    for process in processes
                    if process.exitcode not in (None, 0)
                ]
                if crashed:
                    raise RuntimeError("branch worker exited before returning its result: " + ", ".join(crashed))
                alive = sum(process.is_alive() for process in processes)
                if alive == 0:
                    missing = sorted(set(range(len(jobs))) - set(completed))
                    raise RuntimeError(f"all branch workers exited with incomplete jobs: {missing}")
                print(
                    f"[branch execute] status=waiting started={len(started_jobs)}/{len(jobs)} "
                    f"completed={len(completed)}/{len(jobs)} alive_workers={alive}/{worker_count} "
                    f"elapsed={time.monotonic() - started_at:.1f}s",
                    flush=True,
                )
                continue

            event = message[0]
            if event == "started":
                _, job_index, pid, gpu_id = message
                started_jobs.add(job_index)
                point = jobs[job_index].point
                print(
                    f"[branch execute] status=job-started job={job_index + 1}/{len(jobs)} "
                    f"seed={point.env_seed} point={point.point_type} worker_pid={pid} gpu={gpu_id}",
                    flush=True,
                )
                continue

            _, job_index, rows, error = message
            completed[job_index] = (rows, error)
            print(
                f"[branch execute] status=job-done progress={len(completed)}/{len(jobs)} "
                f"job={job_index + 1} rows={len(rows)} elapsed={time.monotonic() - started_at:.1f}s",
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
    parser.add_argument(
        "--prepare-workers",
        type=int,
        default=min(os.cpu_count() or 1, 96),
        help="CPU processes used to read traces and select branch points; capped to the number of eligible seeds.",
    )
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.prepare_workers < 1:
        raise SystemExit("--prepare-workers must be >= 1")
    protocol_path = repo_path(args.protocol)
    rollout_dir = repo_path(args.rollout_dir)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = read_json(protocol_path)
    tasks = list(args.tasks) if args.tasks else list(protocol["tasks"])
    task_config = protocol.get("task_config", "demo_brace_trace")
    max_points = int(protocol.get("candidate_points_per_pair", 3))
    control_k = int(protocol.get("control_chunks_k", 3))
    continuation_m = int(protocol.get("continuation_seeds_m", 3))
    alpha = float(protocol["acceptance"]["one_sided_alpha"])
    delta = float(protocol["acceptance"]["minimum_advantage_delta"])
    harness_min_rate = float(
        protocol.get("branch_harness", {}).get("harness_sanity", {}).get("candidate_baseline_min_rate", 0.05)
    )

    all_rows: list[dict[str, Any]] = []
    task_summaries: dict[str, Any] = {}
    errors: list[str] = []
    budgets: dict[str, Any] = {}
    harness_valid = True
    harness_invalid_reason: str | None = None

    for task in tasks:
        seeds_file = (
            repo_path(args.seeds_file)
            if args.seeds_file is not None
            else (REPO_ROOT / "experiments" / "brace" / "seeds" / f"{task}_pilot_seeds.json")
        )
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
            prepare_workers=args.prepare_workers,
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
        task_harness_valid, task_harness_reason = evaluate_harness_sanity(rows, min_rate=harness_min_rate)
        summary["harness_valid"] = task_harness_valid
        if not task_harness_valid:
            summary["harness_invalid_reason"] = task_harness_reason
            harness_valid = False
            if harness_invalid_reason is None:
                harness_invalid_reason = f"{task}:{task_harness_reason}"
        task_summaries[task] = summary
        budgets[task] = {
            "environment_transitions": int(sum(int(row.get("transitions", 0)) for row in rows)),
            "accepted_chunks": int(summary["accepted_points"]),
            "optimizer_examples": 0,
        }
        budgets_dir = output_dir / "budgets"
        budgets_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(budgets_dir / f"{task}_pilot.json", budgets[task])

    scientific_passed = bool(task_summaries) and all(summary["passed"] for summary in task_summaries.values())
    passed = harness_valid and scientific_passed and not errors
    summary = {
        "schema_version": 2,
        "protocol_revision": protocol.get("protocol_revision", "2.3"),
        "passed": passed,
        "complete": not errors,
        "harness_valid": harness_valid,
        "tasks": task_summaries,
        "errors": errors,
        "protocol_path": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "git_commit": git_commit(),
        "artifacts": {"checks": "checks.jsonl", "budgets": "budgets/"},
    }
    if budgets:
        write_json_atomic(output_dir / "budgets.json", budgets)
    if harness_invalid_reason is not None:
        summary["harness_invalid_reason"] = harness_invalid_reason
    write_jsonl_atomic(output_dir / "checks.jsonl", all_rows)
    write_json_atomic(output_dir / "summary.json", summary)
    try:
        from experiments.brace.stage_records import emit_stage_record

        emit_stage_record(
            "branch",
            summary=summary,
            summary_path=output_dir / "summary.json",
            tasks=list(args.tasks),
            label=output_dir.name,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to emit stage record: {exc}", file=sys.stderr)
    print(f"Branch collection passed={passed} harness_valid={harness_valid} summary={output_dir / 'summary.json'}")
    if errors:
        print(errors[0], file=sys.stderr)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
