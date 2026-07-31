#!/usr/bin/env python3
"""Deterministic state-prefix replay gate for BRACE.

The rollout HDF5 stores an initial state followed by joint-space waypoints.  This
audit reconstructs the scene from the manifest env seed, replays those waypoints,
and compares robot, end-effector, and manipulated-object state at frozen points.
Missing state is a hard failure: the audit never silently degrades to RGB-only or
robot-only comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import queue as queue_module
import random
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE1_DIR = REPO_ROOT / "experiments" / "phase1"
for import_path in (REPO_ROOT, PHASE1_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


METRIC_TO_THRESHOLD = {
    "joint_max_error": "joint_max_error",
    "end_effector_translation_error": "end_effector_translation_error",
    "end_effector_rotation_error": "end_effector_rotation_error",
    "object_translation_error": "object_translation_error",
    "object_rotation_error": "object_rotation_error",
}


@dataclass(frozen=True)
class Candidate:
    task: str
    env_seed: int
    rollout_id: int
    success: bool
    path: Path


@dataclass
class Episode:
    joints: np.ndarray
    end_effectors: dict[str, np.ndarray]
    objects: dict[str, np.ndarray]

    @property
    def length(self) -> int:
        return int(self.joints.shape[0])


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_env_seeds_from_json(payload: dict[str, Any]) -> list[int]:
    if "seeds" in payload:
        return [int(seed) for seed in payload["seeds"]]
    if "train_rollout" in payload:
        return [int(seed) for seed in payload["train_rollout"]]
    raise ValueError("seeds file must contain 'seeds' or 'train_rollout'")


def iter_manifest_rows(task_dir: Path) -> Iterable[dict[str, Any]]:
    shards = sorted(task_dir.glob("manifest_shard_*_of_*.jsonl"))
    if shards:
        paths = shards
    else:
        canonical = task_dir / "manifest.jsonl"
        paths = [canonical] if canonical.is_file() else []
    seen: set[tuple[int, int]] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (int(row["env_seed"]), int(row["rollout_id"]))
                if key not in seen:
                    seen.add(key)
                    yield row


def collect_candidates(
    task: str,
    rollout_dir: Path,
    *,
    progress_callback: Callable[[int], None] | None = None,
    progress_every: int = 100,
) -> tuple[list[Candidate], list[str]]:
    task_dir = rollout_dir / task
    errors: list[str] = []
    if not task_dir.is_dir():
        return [], [f"missing task rollout directory: {task_dir}"]
    if progress_every < 1:
        raise ValueError("progress_every must be >= 1")

    candidates: list[Candidate] = []
    row_count = 0
    try:
        for row in iter_manifest_rows(task_dir):
            row_count += 1
            if progress_callback is not None and row_count % progress_every == 0:
                progress_callback(row_count)

            success = bool(row.get("success"))
            raw_path = row.get("hdf5_path") if success else row.get("failure_hdf5_path")
            if not raw_path:
                continue
            path = repo_path(raw_path)
            if not path.is_file():
                errors.append(
                    f"missing {'success' if success else 'failure'} HDF5 for "
                    f"seed={row.get('env_seed')} rollout={row.get('rollout_id')}: {path}"
                )
                continue
            candidates.append(
                Candidate(
                    task=task,
                    env_seed=int(row["env_seed"]),
                    rollout_id=int(row["rollout_id"]),
                    success=success,
                    path=path,
                )
            )
    except Exception as exc:
        return [], [f"cannot read manifest: {type(exc).__name__}: {exc}"]
    if row_count == 0:
        return [], [f"no canonical or shard manifest rows under {task_dir}"]
    if progress_callback is not None and row_count % progress_every != 0:
        progress_callback(row_count)
    candidates.sort(key=lambda item: (item.env_seed, item.rollout_id, not item.success))
    return candidates, errors


def select_candidates(
    candidates: list[Candidate],
    count: int,
    seed: int,
    require_both: bool,
) -> tuple[list[Candidate], list[str]]:
    """Deterministically select a near-balanced success/failure sample."""
    successes = [candidate for candidate in candidates if candidate.success]
    failures = [candidate for candidate in candidates if not candidate.success]
    errors: list[str] = []
    if len(candidates) < count:
        errors.append(f"need {count} trajectories, found {len(candidates)} with readable HDF5")
        return [], errors
    if require_both and (not successes or not failures):
        errors.append(
            "audit requires both outcomes, found "
            f"success={len(successes)} failure={len(failures)}"
        )
        return [], errors

    rng = random.Random(seed)
    rng.shuffle(successes)
    rng.shuffle(failures)
    if require_both:
        success_target = min((count + 1) // 2, len(successes))
        failure_target = min(count // 2, len(failures))
        selected = successes[:success_target] + failures[:failure_target]
    else:
        combined = candidates.copy()
        rng.shuffle(combined)
        selected = combined[:count]

    if len(selected) < count:
        selected_keys = {(item.env_seed, item.rollout_id) for item in selected}
        remainder = [
            item for item in successes + failures
            if (item.env_seed, item.rollout_id) not in selected_keys
        ]
        rng.shuffle(remainder)
        selected.extend(remainder[: count - len(selected)])
    selected.sort(key=lambda item: (item.env_seed, item.rollout_id))
    return selected, errors


def _read_group_arrays(group: Any, expected_length: int, label: str) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, dataset in group.items():
        if not hasattr(dataset, "shape"):
            raise ValueError(f"{label}/{name} must be a dataset")
        value = np.asarray(dataset[()], dtype=np.float64)
        if value.ndim != 2 or value.shape != (expected_length, 7):
            raise ValueError(
                f"{label}/{name} has shape {value.shape}, expected ({expected_length}, 7)"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{label}/{name} contains non-finite values")
        arrays[name] = value
    if not arrays:
        raise ValueError(f"{label} contains no pose datasets")
    return arrays


def load_episode(path: Path) -> Episode:
    import h5py

    with h5py.File(path, "r") as root:
        if "joint_action/vector" not in root:
            raise ValueError("missing /joint_action/vector")
        joints = np.asarray(root["joint_action/vector"][()], dtype=np.float64)
        if joints.ndim != 2 or joints.shape[0] < 2:
            raise ValueError(f"/joint_action/vector has invalid shape {joints.shape}")
        if not np.all(np.isfinite(joints)):
            raise ValueError("/joint_action/vector contains non-finite values")
        if "endpose" not in root:
            raise ValueError("missing /endpose state group")
        end_effectors: dict[str, np.ndarray] = {}
        for name in ("left_endpose", "right_endpose"):
            if name not in root["endpose"]:
                raise ValueError(f"/endpose must contain {name}")
            value = np.asarray(root[f"endpose/{name}"][()], dtype=np.float64)
            if value.shape != (joints.shape[0], 7):
                raise ValueError(
                    f"/endpose/{name} has shape {value.shape}, "
                    f"expected ({joints.shape[0]}, 7)"
                )
            if not np.all(np.isfinite(value)):
                raise ValueError(f"/endpose/{name} contains non-finite values")
            end_effectors[name] = value
        if "task_object_pose" not in root:
            raise ValueError(
                "missing /task_object_pose; recollect this rollout with "
                "data_type.task_object_pose=true"
            )
        objects = _read_group_arrays(
            root["task_object_pose"], joints.shape[0], "/task_object_pose"
        )
    return Episode(joints=joints, end_effectors=end_effectors, objects=objects)


def checkpoint_indices(length: int, count: int) -> list[int]:
    if length < 2:
        raise ValueError("an episode needs an initial state and at least one waypoint")
    if count < 1:
        raise ValueError("checkpoints_per_trajectory must be positive")
    available = length - 1
    if available < count:
        raise ValueError(f"episode has only {available} replayable states for {count} checkpoints")
    # Interior quantiles reduce sensitivity to a success-triggered final early return.
    values = np.linspace(1, length - 1, count + 2, dtype=np.int64)[1:-1]
    indices = sorted(set(int(value) for value in values))
    if len(indices) != count:
        indices = [int(value) for value in np.linspace(1, length - 1, count, dtype=np.int64)]
    if len(set(indices)) != count:
        raise ValueError(f"cannot choose {count} unique checkpoints from {length} states")
    return indices


def quaternion_error(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0 or right_norm == 0:
        return math.inf
    dot = float(np.dot(left / left_norm, right / right_norm))
    return float(2.0 * math.acos(min(1.0, abs(dot))))


def pose_errors(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    return (
        float(np.linalg.norm(actual[:3] - expected[:3])),
        quaternion_error(actual[3:7], expected[3:7]),
    )


def current_state(env: Any) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    joints = np.asarray(
        env.robot.get_left_arm_jointState() + env.robot.get_right_arm_jointState(),
        dtype=np.float64,
    )
    end_effectors = {
        "left_endpose": np.asarray(env.get_arm_pose("left"), dtype=np.float64),
        "right_endpose": np.asarray(env.get_arm_pose("right"), dtype=np.float64),
    }
    objects = {
        name: np.concatenate(
            (np.asarray(actor.get_pose().p), np.asarray(actor.get_pose().q))
        ).astype(np.float64)
        for name, actor in env.get_task_object_actors().items()
    }
    return joints, end_effectors, objects


def compare_state(
    actual: tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]],
    episode: Episode,
    index: int,
) -> dict[str, float]:
    return compare_state_detailed(actual, episode, index, include_actor_errors=False)


def compare_state_detailed(
    actual: tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]],
    episode: Episode,
    index: int,
    *,
    include_actor_errors: bool = True,
) -> dict[str, Any]:
    joints, end_effectors, objects = actual
    if joints.shape != episode.joints[index].shape:
        raise ValueError(
            f"replay joint shape {joints.shape} != recorded shape {episode.joints[index].shape}"
        )
    if set(end_effectors) != set(episode.end_effectors):
        raise ValueError("replay and recorded end-effector names differ")
    if set(objects) != set(episode.objects):
        raise ValueError(
            "replay and recorded object names differ: "
            f"replay={sorted(objects)} recorded={sorted(episode.objects)}"
        )

    ee_errors = [
        pose_errors(end_effectors[name], episode.end_effectors[name][index])
        for name in sorted(end_effectors)
    ]
    object_errors = [
        pose_errors(objects[name], episode.objects[name][index])
        for name in sorted(objects)
    ]
    result: dict[str, Any] = {
        "joint_max_error": float(np.max(np.abs(joints - episode.joints[index]))),
        "end_effector_translation_error": max(error[0] for error in ee_errors),
        "end_effector_rotation_error": max(error[1] for error in ee_errors),
        "object_translation_error": max(error[0] for error in object_errors),
        "object_rotation_error": max(error[1] for error in object_errors),
    }
    if include_actor_errors:
        result["actor_errors"] = {
            name: {
                "translation_error": float(translation),
                "rotation_error": float(rotation),
            }
            for name, (translation, rotation) in zip(sorted(objects), object_errors)
        }
    return result


def thresholds_from_protocol(protocol: dict[str, Any]) -> dict[str, float]:
    if protocol.get("status") != "frozen":
        raise ValueError("protocol status must be frozen")
    gate = protocol.get("replay_gate")
    if not isinstance(gate, dict):
        raise ValueError("protocol replay_gate must be an object")
    thresholds: dict[str, float] = {}
    for metric, key in METRIC_TO_THRESHOLD.items():
        value = gate.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"replay_gate.{key} must be numeric")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"replay_gate.{key} must be finite and non-negative")
        thresholds[metric] = float(value)
    pass_rate = gate.get("minimum_pass_rate")
    if isinstance(pass_rate, bool) or not isinstance(pass_rate, (int, float)):
        raise ValueError("replay_gate.minimum_pass_rate must be numeric")
    if not 0 <= float(pass_rate) <= 1:
        raise ValueError("replay_gate.minimum_pass_rate must be in [0, 1]")
    return thresholds


def make_environment(task: str) -> tuple[Any, dict[str, Any]]:
    from common import load_task_args, make_task_env

    return make_task_env(task), load_task_args(task, "demo_clean")


def replay_candidate(
    candidate: Candidate,
    episode: Episode,
    checkpoints: list[int],
    thresholds: dict[str, float],
    environment_factory: Callable[[str], tuple[Any, dict[str, Any]]] = make_environment,
) -> list[dict[str, Any]]:
    env, env_args = environment_factory(candidate.task)
    rows: list[dict[str, Any]] = []
    try:
        run_args = dict(env_args)
        run_args.update(
            {
                "need_plan": False,
                "save_data": False,
                "eval_mode": True,
                "render_freq": 0,
            }
        )
        env.setup_demo(
            now_ep_num=0,
            seed=candidate.env_seed,
            is_test=True,
            **run_args,
        )
        env.step_lim = max(int(env.step_lim), episode.length + 1)
        checkpoint_set = set(checkpoints)
        for index in range(1, max(checkpoints) + 1):
            env.take_action(episode.joints[index])
            if index not in checkpoint_set:
                continue
            errors = compare_state(current_state(env), episode, index)
            metric_passed = {
                metric: value <= thresholds[metric] for metric, value in errors.items()
            }
            rows.append(
                {
                    "task": candidate.task,
                    "env_seed": candidate.env_seed,
                    "rollout_id": candidate.rollout_id,
                    "success": candidate.success,
                    "hdf5_path": str(candidate.path),
                    "checkpoint_index": index,
                    "errors": errors,
                    "metric_passed": metric_passed,
                    "passed": all(metric_passed.values()),
                }
            )
    finally:
        try:
            env.close_env()
        except Exception:
            pass
    return rows


def replay_worker(
    gpu: int,
    jobs: Any,
    results: Any,
) -> None:
    """Replay queued trajectories in a simulator process pinned to one GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    while True:
        job = jobs.get()
        if job is None:
            return
        job_index, candidate, points, thresholds = job
        try:
            episode = load_episode(candidate.path)
            rows = replay_candidate(candidate, episode, points, thresholds)
            results.put((job_index, rows, None))
        except BaseException as exc:
            results.put(
                (
                    job_index,
                    [],
                    f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                )
            )


def replay_candidates_parallel(
    work: list[tuple[Candidate, list[int]]],
    thresholds: dict[str, float],
    gpus: list[int],
) -> tuple[list[dict[str, Any]], list[tuple[Candidate, str]]]:
    """Replay trajectories in stable input order using one process per GPU."""
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
            target=replay_worker,
            args=(gpus[index], job_queues[index], result_queue),
            name=f"replay-audit-gpu-{gpus[index]}",
        )
        for index in range(worker_count)
    ]
    for worker in workers:
        worker.start()

    try:
        for job_index, (candidate, points) in enumerate(work):
            job_queues[job_index % worker_count].put(
                (job_index, candidate, points, thresholds)
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
                        "replay worker exited before returning its result: "
                        + ", ".join(crashed)
                    )
                continue
            completed[job_index] = (rows, error)
            print(
                f"Replay audit trajectory {len(completed)}/{len(work)} completed",
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return None


def build_summary(
    protocol_path: Path,
    protocol: dict[str, Any],
    task_summaries: dict[str, Any],
    checks: list[dict[str, Any]],
    preflight_errors: list[str],
) -> dict[str, Any]:
    gate = protocol["replay_gate"]
    passed_checks = sum(bool(row["passed"]) for row in checks)
    total_checks = len(checks)
    pass_rate = passed_checks / total_checks if total_checks else 0.0
    expected_checks = sum(
        int(task["requested_trajectories"]) * int(task["checkpoints_per_trajectory"])
        for task in task_summaries.values()
    )
    complete = total_checks == expected_checks and not preflight_errors
    passed = complete and pass_rate >= float(gate["minimum_pass_rate"])
    metric_errors = {
        metric: {
            "maximum": max(values),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
        }
        for metric in METRIC_TO_THRESHOLD
        if (values := [float(row["errors"][metric]) for row in checks])
    }
    return {
        "schema_version": 1,
        "passed": passed,
        "complete": complete,
        "pass_rate": pass_rate,
        "minimum_pass_rate": float(gate["minimum_pass_rate"]),
        "passed_checkpoints": passed_checks,
        "total_checkpoints": total_checks,
        "expected_checkpoints": expected_checks,
        "failed_checkpoints": total_checks - passed_checks,
        "protocol_path": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "git_commit": git_commit(),
        "thresholds": thresholds_from_protocol(protocol),
        "metric_errors": metric_errors,
        "tasks": task_summaries,
        "preflight_errors": preflight_errors,
        "artifacts": {
            "checks": "checks.jsonl",
            "failures": "failures.jsonl",
            "diagnostics": "diagnostics.jsonl",
        },
    }


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
        thresholds = thresholds_from_protocol(protocol)
        sampling = protocol["replay_audit"]
        trajectory_count = int(sampling["trajectories_per_task"])
        checkpoint_count = int(sampling["checkpoints_per_trajectory"])
        selection_seed = int(sampling["selection_seed"])
        require_both = bool(sampling["require_success_and_failure"])
        if sampling.get("outcome_sampling") != "near_balanced_without_replacement":
            raise ValueError("unsupported replay_audit.outcome_sampling")
        if sampling.get("checkpoint_rule") != "interior_quartiles":
            raise ValueError("unsupported replay_audit.checkpoint_rule")
        if sampling.get("action_source") != "/joint_action/vector[1:] (next-state joint targets)":
            raise ValueError("unsupported replay_audit.action_source")
        if trajectory_count < 1 or checkpoint_count < 1:
            raise ValueError("frozen replay audit counts must be positive")
    except Exception as exc:
        summary = {
            "schema_version": 1,
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
            "checkpoints_per_trajectory": checkpoint_count,
            "preflight_errors": all_errors,
        }

    # Validate every selected HDF5 before importing or launching the simulator.
    episodes: dict[tuple[str, int, int], Episode] = {}
    for task, selected in selected_by_task.items():
        for candidate in selected:
            try:
                episode = load_episode(candidate.path)
                checkpoint_indices(episode.length, checkpoint_count)
                episodes[(task, candidate.env_seed, candidate.rollout_id)] = episode
            except Exception as exc:
                message = (
                    f"{task}: invalid HDF5 seed={candidate.env_seed} "
                    f"rollout={candidate.rollout_id}: {type(exc).__name__}: {exc}"
                )
                preflight_errors.append(message)
                task_summaries[task]["preflight_errors"].append(message)

    if not preflight_errors:
        work: list[tuple[Candidate, list[int]]] = []
        for task in protocol["tasks"]:
            for candidate in selected_by_task[task]:
                episode = episodes[(task, candidate.env_seed, candidate.rollout_id)]
                points = checkpoint_indices(episode.length, checkpoint_count)
                work.append((candidate, points))
        try:
            checks, replay_errors = replay_candidates_parallel(
                work, thresholds, args.gpus
            )
            for candidate, error in replay_errors:
                message = (
                    f"{candidate.task}: replay failed seed={candidate.env_seed} "
                    f"rollout={candidate.rollout_id}: {error}"
                )
                preflight_errors.append(message)
                task_summaries[candidate.task]["preflight_errors"].append(message)
        except Exception as exc:
            preflight_errors.append(
                f"parallel replay failed: {type(exc).__name__}: {exc}"
            )

    for task, task_summary in task_summaries.items():
        task_checks = [row for row in checks if row["task"] == task]
        task_passed = sum(bool(row["passed"]) for row in task_checks)
        task_summary.update(
            {
                "total_checkpoints": len(task_checks),
                "passed_checkpoints": task_passed,
                "pass_rate": task_passed / len(task_checks) if task_checks else 0.0,
            }
        )

    summary = build_summary(
        protocol_path, protocol, task_summaries, checks, preflight_errors
    )
    write_jsonl_atomic(output_dir / "checks.jsonl", checks)
    write_jsonl_atomic(
        output_dir / "failures.jsonl",
        [row for row in checks if not row["passed"]],
    )
    write_jsonl_atomic(
        output_dir / "diagnostics.jsonl",
        [{"error": error} for error in preflight_errors],
    )
    write_json_atomic(output_dir / "summary.json", summary)
    print(
        f"Replay audit passed={summary['passed']} complete={summary['complete']} "
        f"passed_checkpoints={summary['passed_checkpoints']}/{summary['total_checkpoints']} "
        f"completed_checkpoints={summary['total_checkpoints']}/{summary['expected_checkpoints']} "
        f"summary={output_dir / 'summary.json'}"
    )
    if preflight_errors:
        print(preflight_errors[0], file=sys.stderr)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
