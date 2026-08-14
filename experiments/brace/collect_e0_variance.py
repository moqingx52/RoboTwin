#!/usr/bin/env python3
"""E0 variance gate: same-state action resampling to estimate Var_a[Q(s,a)].

For each branch point s (from mixed-outcome traced seeds), sample A action
chunks a_1..a_A from the frozen policy at the restored state with distinct
diffusion noise seeds, then execute each chunk R times with independent
continuation seeds. The one-way random-effects decomposition of the R
Bernoulli outcomes per action yields an unbiased estimate of
sigma^2_Q = Var_a[Q(s,a)], which bounds the exact BRACE-RW improvement
Delta_c(s) = c * Var_a[Q] / (1 + c V(s)).

Continuation seeds are independent across actions (no common random numbers):
CRN would correlate outcomes across actions and bias S^2_between.

Outputs: checks.jsonl (one row per rollout), chunks/*.npz (sampled action
chunks, reusable as the BRACE-RW dataset source), summary.json.
Analysis lives in analyze_e0_variance.py.
"""

from __future__ import annotations

import argparse
import hashlib
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

from experiments.brace.collect_branches import (
    BranchPoint,
    _run_branch_episode,
    select_branch_points,
)
from experiments.brace.control_trace import build_branch_context, load_brace_trace
from experiments.brace.replay_audit import (
    collect_candidates,
    file_sha256,
    git_commit,
    read_json,
    repo_path,
    write_json_atomic,
    write_jsonl_atomic,
)
from experiments.brace.replay_audit_v2 import (
    job_worker_index,
    resolve_worker_count,
    worker_gpu_assignments,
)


@dataclass(frozen=True)
class E0Job:
    point: BranchPoint
    action_seeds: tuple[int, ...]
    continuation_seeds: tuple[tuple[int, ...], ...]  # [action_index][continuation_index]


def derive_seed(namespace: str, *parts: Any) -> int:
    """Deterministic, collision-resistant 31-bit seed from a namespace + parts."""
    digest = hashlib.sha256(("e0:" + namespace + ":" + ":".join(str(p) for p in parts)).encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


def build_e0_jobs(
    task: str,
    rollout_dir: Path,
    env_seeds: list[int],
    *,
    max_points: int,
    action_count: int,
    continuation_count: int,
    selection_seed: int,
) -> list[E0Job]:
    candidates, errors = collect_candidates(task, rollout_dir)
    for error in errors:
        print(f"[e0 prepare] task={task} warning={error}", flush=True)

    by_seed: dict[int, dict[bool, list[Any]]] = {}
    for candidate in candidates:
        by_seed.setdefault(candidate.env_seed, {}).setdefault(candidate.success, []).append(candidate)

    jobs: list[E0Job] = []
    for seed_index, env_seed in enumerate(env_seeds):
        seed_candidates = by_seed.get(env_seed, {})
        successes = seed_candidates.get(True, [])
        failures = seed_candidates.get(False, [])
        if not successes or not failures:
            print(
                f"[e0 prepare] task={task} seed={env_seed} status=skipped "
                f"successes={len(successes)} failures={len(failures)} (need mixed outcome)",
                flush=True,
            )
            continue
        success = sorted(successes, key=lambda item: item.rollout_id)[0]
        failure = sorted(failures, key=lambda item: item.rollout_id)[0]
        rng = random.Random(selection_seed + seed_index)
        points = select_branch_points(
            task, env_seed, success, failure, max_points=max_points, rng=rng
        )
        for point in points:
            action_seeds = tuple(
                derive_seed("action", task, env_seed, point.snapshot_id, a) for a in range(action_count)
            )
            continuation_seeds = tuple(
                tuple(
                    derive_seed("continuation", task, env_seed, point.snapshot_id, a, r)
                    for r in range(continuation_count)
                )
                for a in range(action_count)
            )
            jobs.append(E0Job(point=point, action_seeds=action_seeds, continuation_seeds=continuation_seeds))
    return jobs


def _sample_action_chunks(
    env,
    *,
    snapshot: dict[str, Any],
    replay_steps: list[dict[str, Any]],
    runtime_state: dict[str, Any],
    action_seeds: tuple[int, ...],
    model,
    encode_obs,
) -> np.ndarray:
    """Sample len(action_seeds) chunks from the frozen policy at the restored state.

    The env is never stepped, so all chunks condition on the identical state;
    only the diffusion noise seed differs.
    """
    import torch

    env.restore_branch_snapshot(snapshot)
    for step in replay_steps:
        env.replay_control_step(step)
    env.apply_branch_runtime_state(runtime_state)
    observation = env.get_obs()
    obs = encode_obs(observation)

    chunks = []
    for action_seed in action_seeds:
        model.reset_obs()
        generator = torch.Generator(device="cuda:0")
        generator.manual_seed(int(action_seed))
        model.set_generator(generator)
        actions = model.get_action(obs)
        chunks.append(np.asarray(actions, dtype=np.float64))
    return np.stack(chunks, axis=0)


def run_e0_job(
    job: E0Job,
    rollout_dir: Path,
    chunks_dir: Path,
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

    env = make_task_env(task)
    try:
        env.setup_demo(now_ep_num=0, seed=job.point.env_seed, is_test=True, **run_args)
        chunks = _sample_action_chunks(
            env,
            snapshot=snapshot,
            replay_steps=branch_context.replay_steps,
            runtime_state=branch_context.runtime_state,
            action_seeds=job.action_seeds,
            model=model,
            encode_obs=encode_obs,
        )
    finally:
        try:
            env.close_env()
        except Exception:
            pass

    chunks_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        chunks_dir / f"{task}_seed{job.point.env_seed}_snap{job.point.snapshot_id}.npz",
        actions=chunks,
        action_seeds=np.asarray(job.action_seeds, dtype=np.int64),
        env_seed=np.int64(job.point.env_seed),
        snapshot_id=np.int64(job.point.snapshot_id),
        branch_chunk_index=np.int64(job.point.branch_chunk_index),
        point_type=np.str_(job.point.point_type),
    )

    base_fields = {
        **job.point.__dict__,
        "boundary_physics_step": int(branch_context.boundary_physics_step),
    }
    rows: list[dict[str, Any]] = []
    for action_index, action_seed in enumerate(job.action_seeds):
        for continuation_index, continuation_seed in enumerate(job.continuation_seeds[action_index]):
            env = make_task_env(task)
            try:
                env.setup_demo(now_ep_num=0, seed=job.point.env_seed, is_test=True, **run_args)
                result = _run_branch_episode(
                    env,
                    snapshot=snapshot,
                    replay_steps=branch_context.replay_steps,
                    runtime_state=branch_context.runtime_state,
                    chunk_actions=chunks[action_index],
                    continuation_seed=continuation_seed,
                    model=model,
                    encode_obs=encode_obs,
                )
                rows.append(
                    {
                        **base_fields,
                        "action_index": action_index,
                        "action_seed": int(action_seed),
                        "continuation_index": continuation_index,
                        "continuation_seed": int(continuation_seed),
                        **result,
                    }
                )
            finally:
                try:
                    env.close_env()
                except Exception:
                    pass
    return rows


def e0_worker(gpu_id, jobs, results, rollout_dir, chunks_dir, task_config, model_args) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    while True:
        item = jobs.get()
        if item is None:
            return
        job_index, job = item
        results.put(("started", job_index, os.getpid(), gpu_id))
        try:
            rows = run_e0_job(job, rollout_dir, chunks_dir, task_config=task_config, model_args=model_args)
            results.put(("done", job_index, rows, None))
        except BaseException as exc:
            results.put(("done", job_index, [], f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))


def run_e0_jobs_parallel(
    jobs: list[E0Job],
    rollout_dir: Path,
    chunks_dir: Path,
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
        work_size=len(jobs), workers=workers, workers_per_gpu=workers_per_gpu, gpu_ids=gpu_ids
    )
    assignments = worker_gpu_assignments(gpu_ids, workers_per_gpu)[:worker_count]
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    job_queues = [context.Queue() for _ in range(worker_count)]
    processes = [
        context.Process(
            target=e0_worker,
            args=(gpu_id, job_queues[worker_index], result_queue, rollout_dir, chunks_dir, task_config, model_args),
            name=f"e0-variance-gpu-{gpu_id}-worker-{worker_index}",
        )
        for worker_index, gpu_id in assignments
    ]
    started_at = time.monotonic()
    print(
        f"[e0 execute] status=launching jobs={len(jobs)} workers={worker_count} "
        f"workers_per_gpu={workers_per_gpu} gpus={gpu_ids}",
        flush=True,
    )
    for process in processes:
        process.start()
    try:
        for job_index, job in enumerate(jobs):
            job_queues[job_worker_index(job_index, worker_count)].put((job_index, job))
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
                    raise RuntimeError("e0 worker exited before returning its result: " + ", ".join(crashed))
                alive = sum(process.is_alive() for process in processes)
                if alive == 0:
                    missing = sorted(set(range(len(jobs))) - set(completed))
                    raise RuntimeError(f"all e0 workers exited with incomplete jobs: {missing}")
                print(
                    f"[e0 execute] status=waiting started={len(started_jobs)}/{len(jobs)} "
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
                    f"[e0 execute] status=job-started job={job_index + 1}/{len(jobs)} "
                    f"seed={point.env_seed} point={point.point_type} worker_pid={pid} gpu={gpu_id}",
                    flush=True,
                )
                continue

            _, job_index, rows, error = message
            completed[job_index] = (rows, error)
            print(
                f"[e0 execute] status=job-done progress={len(completed)}/{len(jobs)} "
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
    e0 = protocol["e0_variance_gate"]
    max_points = int(e0["points_per_seed"])
    action_count = int(e0["actions_per_state"])
    continuation_count = int(e0["continuations_per_action"])
    selection_seed = int(e0.get("selection_seed", 0))

    all_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    task_counts: dict[str, Any] = {}
    for task in tasks:
        if args.seeds_file is not None:
            seeds_file = repo_path(args.seeds_file)
        elif e0.get("seeds_file"):
            seeds_file = repo_path(Path(str(e0["seeds_file"]).format(task=task)))
        else:
            seeds_file = REPO_ROOT / "experiments" / "brace" / "seeds" / f"{task}_pilot_seeds.json"
        if not seeds_file.is_file():
            errors.append(f"{task}: missing seeds file {seeds_file}")
            continue
        seeds_payload = read_json(seeds_file)
        seeds_key = str(e0.get("seeds_key", "seeds"))
        if seeds_key not in seeds_payload:
            errors.append(f"{task}: seeds file {seeds_file} has no key {seeds_key!r}")
            continue
        env_seeds = [int(seed) for seed in seeds_payload[seeds_key]]
        jobs = build_e0_jobs(
            task,
            rollout_dir,
            env_seeds,
            max_points=max_points,
            action_count=action_count,
            continuation_count=continuation_count,
            selection_seed=selection_seed,
        )
        expected_rollouts = len(jobs) * action_count * continuation_count
        print(
            f"[e0] task={task} points={len(jobs)} actions_per_state={action_count} "
            f"continuations={continuation_count} expected_rollouts={expected_rollouts}",
            flush=True,
        )
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
        rows, job_errors = run_e0_jobs_parallel(
            jobs,
            rollout_dir,
            output_dir / "chunks",
            workers=workers,
            gpu_ids=args.gpus,
            workers_per_gpu=args.workers_per_gpu,
            task_config=task_config,
            model_args=model_args,
        )
        errors.extend(job_errors)
        all_rows.extend(rows)
        task_counts[task] = {
            "points": len(jobs),
            "rollouts": len(rows),
            "expected_rollouts": expected_rollouts,
            "environment_transitions": int(sum(int(row.get("transitions", 0)) for row in rows)),
        }

    complete = not errors and all(
        counts["rollouts"] == counts["expected_rollouts"] for counts in task_counts.values()
    )
    summary = {
        "schema_version": 1,
        "stage": "e0_variance_gate",
        "protocol_revision": protocol.get("protocol_revision"),
        "complete": bool(complete),
        "tasks": task_counts,
        "actions_per_state": action_count,
        "continuations_per_action": continuation_count,
        "errors": errors,
        "protocol_path": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "git_commit": git_commit(),
        "artifacts": {"checks": "checks.jsonl", "chunks": "chunks/", "analysis": "analysis.json"},
    }
    write_jsonl_atomic(output_dir / "checks.jsonl", all_rows)
    write_json_atomic(output_dir / "summary.json", summary)
    try:
        from experiments.brace.stage_records import emit_stage_record

        emit_stage_record(
            "e0_variance_gate",
            summary=summary,
            summary_path=output_dir / "summary.json",
            tasks=tasks,
            label=output_dir.name,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to emit stage record: {exc}", file=sys.stderr)
    print(f"E0 collection complete={complete} summary={output_dir / 'summary.json'}")
    if errors:
        print(errors[0], file=sys.stderr)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
