#!/usr/bin/env python3
"""Resumable Phase 2 scheduler.

Each GPU is in exactly one mode:
  * one DP fine-tuning process, or
  * up to N DP evaluation shard processes.
"""

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE1 = REPO_ROOT / "experiments" / "phase1"
PHASE2 = REPO_ROOT / "experiments" / "phase2"
DP_DIR = REPO_ROOT / "policy" / "DP"

VARIANT_CONFIGS = {
    "expert_only": {"dataset_suffix": "expert_only", "expert_ratio": "none"},
    "uniform_mixed": {"dataset_suffix": "success", "expert_ratio": "none"},
    "anchored_70": {"dataset_suffix": "success", "expert_ratio": "0.7"},
    "anchored_50": {"dataset_suffix": "success", "expert_ratio": "0.5"},
    "anchored_70_weighted": {
        "dataset_suffix": "difficulty_weighted",
        "expert_ratio": "0.7",
    },
}


def now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def artifact_is_complete(job):
    path = Path(job["artifact"])
    if job["kind"] == "train":
        marker = path.with_suffix(path.suffix + ".complete")
        return path.is_file() and path.stat().st_size > 0 and marker.is_file()
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
        progress = payload.get("progress", {})
        return bool(progress.get("complete")) and int(progress.get("completed_episodes", -1)) == len(
            payload.get("rows", [])
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def merged_result_is_complete(path, task, variant, ckpt_path, expected_episodes):
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
        keys = {
            (row["split"], int(row["env_seed"]), int(row["repeat"]))
            for row in payload["rows"]
        }
        return (
            payload.get("task_name") == task
            and payload.get("variant") == variant
            and payload.get("ckpt_path") == str(ckpt_path)
            and len(keys) == expected_episodes
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def compute_expert_steps(task, batch_size):
    import zarr

    path = DP_DIR / "data_phase1_200" / f"{task}-expert_only.zarr"
    root = zarr.open(str(path), mode="r")
    ends = root["meta/episode_ends"][:]
    if len(ends) < 2:
        raise RuntimeError(f"Not enough episodes in {path}")
    steps = int(ends[-2]) // batch_size
    if steps <= 0:
        raise RuntimeError(f"Not enough samples in {path}")
    return steps


class Scheduler:
    def __init__(self, args):
        self.args = args
        self.gpus = args.gpus
        self.eval_shards = args.eval_shards or len(self.gpus) * args.eval_per_gpu
        self.eval_dir = PHASE2 / "eval_results"
        self.log_dir = PHASE2 / "logs"
        self.figure_dir = PHASE2 / "figures"
        self.state_path = args.state_path
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.processes = {}
        self.stopping = False
        self.config = {
            "tasks": args.tasks,
            "variants": args.variants,
            "train_seeds": args.train_seeds,
            "gpus": args.gpus,
            "eval_per_gpu": args.eval_per_gpu,
            "eval_shards": self.eval_shards,
            "epochs": args.epochs,
            "checkpoint_every": args.checkpoint_every,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_retries": args.max_retries,
            "expected_eval_episodes": args.expected_eval_episodes,
            "max_train_gpus": args.max_train_gpus,
            "retry_backoff": args.retry_backoff,
        }
        self.state = self._load_or_create_state()
        self._prepare_base_results()
        self._refresh_from_artifacts()
        self._save_state()

    def _load_or_create_state(self):
        if self.state_path.exists():
            state = read_json(self.state_path)
            if state.get("config") != self.config:
                raise RuntimeError(
                    f"State config differs from this run: {self.state_path}. "
                    "Use the original configuration or a different PHASE2_STATE_PATH."
                )
            if self.args.retry_failed:
                for job in state["jobs"].values():
                    if job["status"] == "failed":
                        job["status"] = "pending"
                        job["attempts"] = 0
                        job["message"] = "Manually re-queued by --retry-failed"
                for group in state["groups"].values():
                    if group["status"] == "failed":
                        group["status"] = "pending"
                        group["attempts"] = 0
            live_pids = []
            for job in state["jobs"].values():
                if job["status"] == "running":
                    pid = job.get("pid")
                    if pid:
                        try:
                            os.kill(int(pid), 0)
                            live_pids.append((job["id"], int(pid)))
                        except ProcessLookupError:
                            pass
                        except PermissionError:
                            live_pids.append((job["id"], int(pid)))
                    job["status"] = "pending"
                    job["pid"] = None
                    job["gpu"] = None
                    job["attempts"] = max(0, int(job.get("attempts", 0)) - 1)
                    job["message"] = "Recovered after scheduler interruption"
            if live_pids:
                details = ", ".join(f"{job_id} PID={pid}" for job_id, pid in live_pids)
                raise RuntimeError(
                    "Previous scheduler children are still alive; refusing duplicate launch. "
                    f"Stop or wait for them first: {details}"
                )
            state["status"] = "running"
            state["resumed_at"] = now()
            return state

        jobs = {}
        groups = {}
        expert_steps = {task: compute_expert_steps(task, self.args.batch_size) for task in self.args.tasks}
        for seed in self.args.train_seeds:
            for task in self.args.tasks:
                for variant in self.args.variants:
                    cfg = VARIANT_CONFIGS[variant]
                    train_id = f"train:{task}:{variant}:seed{seed}"
                    checkpoint = (
                        DP_DIR
                        / "checkpoints"
                        / f"{task}-phase2-{variant}-{seed}"
                        / f"{self.args.epochs}.ckpt"
                    )
                    jobs[train_id] = {
                        "id": train_id,
                        "kind": "train",
                        "task": task,
                        "variant": variant,
                        "train_seed": seed,
                        "status": "pending",
                        "gpu": None,
                        "pid": None,
                        "attempts": 0,
                        "artifact": str(checkpoint),
                        "log": str(self.log_dir / f"finetune_{task}_{variant}_seed{seed}.log"),
                        "command": [
                            "bash",
                            str(PHASE2 / "finetune.sh"),
                            task,
                            variant,
                            "{gpu}",
                            str(seed),
                            str(self.args.epochs),
                            "14",
                            str(expert_steps[task]),
                            str(self.args.learning_rate),
                            cfg["expert_ratio"],
                            cfg["dataset_suffix"],
                            str(self.args.checkpoint_every),
                        ],
                    }
                    group_id = f"eval:{task}:{variant}:seed{seed}"
                    group_dir = self.eval_dir / f"train_seed_{seed}"
                    groups[group_id] = {
                        "id": group_id,
                        "task": task,
                        "variant": variant,
                        "train_seed": seed,
                        "dependency": train_id,
                        "status": "pending",
                        "attempts": 0,
                        "output_dir": str(group_dir),
                        "artifact": str(group_dir / task / f"{variant}.json"),
                        "checkpoint": str(checkpoint),
                    }
                    for shard in range(self.eval_shards):
                        job_id = f"{group_id}:shard{shard:02d}"
                        jobs[job_id] = {
                            "id": job_id,
                            "kind": "eval",
                            "task": task,
                            "variant": variant,
                            "train_seed": seed,
                            "shard": shard,
                            "num_shards": self.eval_shards,
                            "dependency": train_id,
                            "group": group_id,
                            "status": "pending",
                            "gpu": None,
                            "pid": None,
                            "attempts": 0,
                            "artifact": str(
                                group_dir
                                / task
                                / f"{variant}_shard_{shard:02d}_of_{self.eval_shards:02d}.json"
                            ),
                            "log": str(
                                self.log_dir
                                / f"eval_{task}_{variant}_seed{seed}_shard{shard:02d}.log"
                            ),
                            "command": [
                                "python",
                                str(PHASE1 / "eval_per_seed.py"),
                                "--task",
                                task,
                                "--task-config",
                                "demo_clean",
                                "--variant",
                                variant,
                                "--output-dir",
                                str(group_dir),
                                "--shard-id",
                                str(shard),
                                "--num-shards",
                                str(self.eval_shards),
                                "--resume",
                                "--ckpt-path",
                                str(checkpoint),
                                "--rollout-dir",
                                str(PHASE1 / "rollouts_200"),
                                "--hard-seeds-file",
                                str(PHASE1 / "eval_results_200" / "hard_eval_seeds" / f"{task}.json"),
                                "--policy-seed-offset",
                                "1000",
                            ],
                        }
        return {
            "version": 1,
            "status": "running",
            "created_at": now(),
            "updated_at": now(),
            "config": self.config,
            "jobs": jobs,
            "groups": groups,
            "events": [],
            "gpu_assignments": {},
        }

    def _event(self, message):
        self.state["events"].append({"time": now(), "message": message})
        self.state["events"] = self.state["events"][-200:]
        print(f"[{now()}] {message}", flush=True)

    def _prepare_base_results(self):
        for seed in self.args.train_seeds:
            for task in self.args.tasks:
                src = PHASE1 / "eval_results_200" / f"train_seed_{seed}" / task / "base.json"
                dst = self.eval_dir / f"train_seed_{seed}" / task / "base.json"
                if not src.is_file():
                    raise FileNotFoundError(f"Missing Phase 1 base evaluation: {src}")
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(src, dst)

    def _refresh_from_artifacts(self):
        for group in self.state["groups"].values():
            merged = Path(group["artifact"])
            if merged_result_is_complete(
                merged,
                group["task"],
                group["variant"],
                Path(group["checkpoint"]),
                self.args.expected_eval_episodes,
            ):
                group["status"] = "completed"
                for job in self.state["jobs"].values():
                    if job.get("group") == group["id"]:
                        job["status"] = "completed"
                        job["pid"] = None
                        job["gpu"] = None
        for job in self.state["jobs"].values():
            if (
                job["id"] not in self.processes
                and job["status"] != "completed"
                and artifact_is_complete(job)
            ):
                job["status"] = "completed"
                job["pid"] = None
                job["gpu"] = None

    def _gpu_assignments(self):
        result = {
            str(gpu): {"mode": "idle", "train": None, "eval": []}
            for gpu in self.gpus
        }
        for job_id, info in self.processes.items():
            gpu = str(info["gpu"])
            if info["job"]["kind"] == "train":
                result[gpu]["mode"] = "train"
                result[gpu]["train"] = job_id
            else:
                result[gpu]["mode"] = "eval"
                result[gpu]["eval"].append(job_id)
        return result

    def _save_state(self):
        self.state["updated_at"] = now()
        self.state["gpu_assignments"] = self._gpu_assignments()
        self._snapshot_progress()
        counts = {}
        for job in self.state["jobs"].values():
            counts[job["status"]] = counts.get(job["status"], 0) + 1
        self.state["job_counts"] = counts
        atomic_write_json(self.state_path, self.state)

    def _snapshot_progress(self):
        for job in self.state["jobs"].values():
            if job["kind"] == "train":
                checkpoint_dir = Path(job["artifact"]).parent
                completed_epochs = 0
                if checkpoint_dir.is_dir():
                    for path in checkpoint_dir.glob("*.ckpt"):
                        try:
                            completed_epochs = max(completed_epochs, int(path.stem))
                        except ValueError:
                            continue
                job["progress"] = {
                    "completed_epochs": completed_epochs,
                    "total_epochs": self.args.epochs,
                }
                continue

            completed_episodes = 0
            path = Path(job["artifact"])
            if path.is_file():
                try:
                    payload = read_json(path)
                    completed_episodes = len(payload.get("rows", []))
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            shard = int(job["shard"])
            target = max(
                0,
                (self.args.expected_eval_episodes + self.eval_shards - 1 - shard)
                // self.eval_shards,
            )
            job["progress"] = {
                "completed_episodes": completed_episodes,
                "total_episodes": target,
            }

    def _start_job(self, job, gpu):
        command = [str(gpu) if token == "{gpu}" else token for token in job["command"]]
        log_path = Path(job["log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a", encoding="utf-8")
        log_file.write(f"\n===== attempt {job['attempts'] + 1} started {now()} on GPU {gpu} =====\n")
        log_file.flush()
        env = os.environ.copy()
        if job["kind"] == "eval":
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        proc = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        job["status"] = "running"
        job["gpu"] = gpu
        job["pid"] = proc.pid
        job["attempts"] += 1
        job["started_at"] = now()
        job["message"] = ""
        self.processes[job["id"]] = {
            "process": proc,
            "log_file": log_file,
            "gpu": gpu,
            "job": job,
        }
        self._event(f"Started {job['id']} on GPU {gpu}, PID {proc.pid}")

    def _finish_process(self, job_id, returncode):
        info = self.processes.pop(job_id)
        info["log_file"].close()
        job = info["job"]
        job["pid"] = None
        job["gpu"] = None
        job["finished_at"] = now()
        if returncode == 0 and artifact_is_complete(job):
            job["status"] = "completed"
            job["message"] = "artifact verified"
            self._event(f"Completed {job_id}")
        elif self.stopping:
            job["status"] = "pending"
            job["attempts"] = max(0, int(job["attempts"]) - 1)
            job["message"] = "stopped with scheduler; resumable artifact retained"
        elif job["attempts"] < self.args.max_retries:
            job["status"] = "pending"
            job["not_before"] = time.time() + self.args.retry_backoff * job["attempts"]
            job["message"] = f"exit={returncode}; queued for retry"
            self._event(f"Retrying {job_id} after exit {returncode}")
        else:
            job["status"] = "failed"
            job["message"] = f"exit={returncode}; retry limit reached"
            self._event(f"FAILED {job_id} after {job['attempts']} attempts")

    def _poll(self):
        for job_id, info in list(self.processes.items()):
            rc = info["process"].poll()
            if rc is not None:
                self._finish_process(job_id, rc)

    def _dependency_complete(self, job):
        dependency = job.get("dependency")
        return not dependency or self.state["jobs"][dependency]["status"] == "completed"

    def _pending_train(self):
        return [
            job
            for job in self.state["jobs"].values()
            if job["kind"] == "train"
            and job["status"] == "pending"
            and float(job.get("not_before", 0)) <= time.time()
        ]

    def _pending_eval(self):
        return [
            job
            for job in self.state["jobs"].values()
            if job["kind"] == "eval"
            and job["status"] == "pending"
            and self._dependency_complete(job)
            and self.state["groups"][job["group"]]["status"] != "completed"
            and float(job.get("not_before", 0)) <= time.time()
        ]

    def _schedule(self):
        pending_train = self._pending_train()
        pending_eval = self._pending_eval()
        assignments = self._gpu_assignments()
        active_train = sum(1 for value in assignments.values() if value["mode"] == "train")
        for gpu in self.gpus:
            current = assignments[str(gpu)]
            if current["mode"] == "train":
                continue
            if current["mode"] == "eval":
                free = self.args.eval_per_gpu - len(current["eval"])
                for _ in range(min(free, len(pending_eval))):
                    self._start_job(pending_eval.pop(0), gpu)
                continue

            # A completely idle GPU prefers training. Once the train queue is
            # exhausted, it switches to evaluation mode with up to N workers.
            if pending_train and active_train < self.args.max_train_gpus:
                self._start_job(pending_train.pop(0), gpu)
                active_train += 1
            else:
                for _ in range(min(self.args.eval_per_gpu, len(pending_eval))):
                    self._start_job(pending_eval.pop(0), gpu)

    def _merge_ready_groups(self):
        for group in self.state["groups"].values():
            if group["status"] == "completed":
                continue
            shard_jobs = [
                job for job in self.state["jobs"].values() if job.get("group") == group["id"]
            ]
            if not shard_jobs or any(job["status"] != "completed" for job in shard_jobs):
                continue
            command = [
                "python",
                str(PHASE1 / "merge_eval_shards.py"),
                "--task",
                group["task"],
                "--task-config",
                "demo_clean",
                "--variant",
                group["variant"],
                "--output-dir",
                group["output_dir"],
                "--num-shards",
                str(self.eval_shards),
            ]
            log_path = self.log_dir / f"merge_{group['task']}_{group['variant']}_seed{group['train_seed']}.log"
            with log_path.open("a", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            group["attempts"] += 1
            if result.returncode == 0 and merged_result_is_complete(
                Path(group["artifact"]),
                group["task"],
                group["variant"],
                Path(group["checkpoint"]),
                self.args.expected_eval_episodes,
            ):
                group["status"] = "completed"
                group["completed_at"] = now()
                self._event(f"Merged {group['id']}")
            elif group["attempts"] >= self.args.max_retries:
                group["status"] = "failed"
                self._event(f"FAILED merge {group['id']}")

    def _has_terminal_failure(self):
        return any(job["status"] == "failed" for job in self.state["jobs"].values()) or any(
            group["status"] == "failed" for group in self.state["groups"].values()
        )

    def _all_complete(self):
        return all(job["status"] == "completed" for job in self.state["jobs"].values()) and all(
            group["status"] == "completed" for group in self.state["groups"].values()
        )

    def _aggregate(self):
        command = [
            "python",
            str(PHASE2 / "aggregate_eval.py"),
            "--eval-dir",
            str(self.eval_dir),
            "--train-seeds",
            *[str(seed) for seed in self.args.train_seeds],
            "--tasks",
            *self.args.tasks,
            "--variants",
            *self.args.variants,
            "--output",
            str(self.eval_dir / "summary.json"),
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        for seed in self.args.train_seeds:
            subprocess.run(
                [
                    "python",
                    str(PHASE1 / "plot_diagnostics.py"),
                    "--tasks",
                    *self.args.tasks,
                    "--variants",
                    "base",
                    *self.args.variants,
                    "--rollout-dir",
                    str(PHASE1 / "rollouts_200"),
                    "--eval-dir",
                    str(self.eval_dir / f"train_seed_{seed}"),
                    "--output-dir",
                    str(self.figure_dir / f"train_seed_{seed}"),
                ],
                cwd=REPO_ROOT,
                check=True,
            )

    def stop(self, final_status="interrupted"):
        if self.stopping:
            return
        self.stopping = True
        self._event("Stopping scheduler and child jobs; checkpoints are retained")
        for info in self.processes.values():
            try:
                os.killpg(info["process"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.time() + 20
        while self.processes and time.time() < deadline:
            self._poll()
            time.sleep(0.5)
        for info in self.processes.values():
            try:
                os.killpg(info["process"].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        while self.processes:
            self._poll()
            time.sleep(0.1)
        self.state["status"] = final_status
        self._save_state()

    def run(self):
        while not self.stopping:
            self._poll()
            self._refresh_from_artifacts()
            self._merge_ready_groups()
            if self._has_terminal_failure():
                self.state["status"] = "failed"
                self._save_state()
                raise RuntimeError(f"Phase 2 has terminal job failures; inspect {self.state_path}")
            if self._all_complete():
                self._aggregate()
                self.state["status"] = "completed"
                self.state["completed_at"] = now()
                self._event("Phase 2 completed")
                self._save_state()
                return
            self._schedule()
            self._save_state()
            time.sleep(self.args.poll_interval)


def parse_args():
    parser = argparse.ArgumentParser(description="Run resumable Phase 2 training and evaluation.")
    parser.add_argument("--tasks", nargs="+", default=["place_container_plate", "dump_bin_bigbin"])
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANT_CONFIGS), default=list(VARIANT_CONFIGS))
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--eval-per-gpu", type=int, default=3)
    parser.add_argument("--eval-shards", type=int)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=60.0)
    parser.add_argument("--max-train-gpus", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--expected-eval-episodes", type=int, default=760)
    parser.add_argument("--state-path", type=Path, default=PHASE2 / "run_state.json")
    args = parser.parse_args()
    if args.eval_per_gpu < 1:
        parser.error("--eval-per-gpu must be at least 1")
    if args.max_train_gpus is None:
        args.max_train_gpus = len(args.gpus)
    if not 1 <= args.max_train_gpus <= len(args.gpus):
        parser.error("--max-train-gpus must be between 1 and the number of configured GPUs")
    if args.epochs < 1 or args.checkpoint_every < 1:
        parser.error("--epochs and --checkpoint-every must be positive")
    return args


def main():
    args = parse_args()
    args.state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.state_path.with_suffix(args.state_path.suffix + ".lock")
    lock_file = lock_path.open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"Another Phase 2 scheduler owns {lock_path}")

    scheduler = Scheduler(args)

    def handle_signal(signum, _frame):
        print(f"Received signal {signum}", flush=True)
        scheduler.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    try:
        scheduler.run()
    except BaseException:
        scheduler.stop(final_status="failed")
        raise


if __name__ == "__main__":
    main()
