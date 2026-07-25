#!/usr/bin/env python3
"""Resumable Phase 2b scheduler with staged diagnose/screen/full/confirm flows."""

import argparse
import copy
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    CANDIDATE_CONFIGS,
    DEFAULT_GPUS,
    DEFAULT_TASKS,
    DIAGNOSE_VARIANTS,
    PHASE1,
    PHASE2B,
    REPO_ROOT,
    SCREEN_CANDIDATES,
    SCREEN_EPOCHS,
    atomic_write_json,
    base_eval_path,
    compute_expert_steps,
    diagnose_ckpt_path,
    hard_seeds_file,
    now,
    phase2b_checkpoint_dir,
    read_json,
)
from promotion import (
    evaluate_diagnose,
    evaluate_full,
    evaluate_screen,
    load_base_metrics,
)


def artifact_is_complete(job):
    path = Path(job["artifact"])
    if job["kind"] == "train":
        marker = path.with_suffix(path.suffix + ".complete")
        return path.is_file() and path.stat().st_size > 0 and marker.is_file()
    if job["kind"] in {"prep", "gate"}:
        return path.is_file()
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


class Scheduler:
    def __init__(self, args):
        self.args = args
        self.stage = args.stage
        self.gpus = args.gpus
        self.eval_shards = args.eval_shards or len(self.gpus) * args.eval_per_gpu
        self.eval_dir = PHASE2B / "eval_results"
        self.log_dir = PHASE2B / "logs"
        self.figure_dir = PHASE2B / "figures"
        self.state_path = args.state_path
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.processes = {}
        self.stopping = False
        self.config = self._build_config()
        self.state = self._load_or_create_state()
        self._refresh_from_artifacts()
        self._save_state()

    def _build_config(self):
        return {
            "stage": self.stage,
            "tasks": self.args.tasks,
            "train_seeds": self.args.train_seeds,
            "gpus": self.gpus,
            "eval_per_gpu": self.args.eval_per_gpu,
            "eval_shards": self.eval_shards,
            "epochs": self.args.epochs,
            "checkpoint_every": self.args.checkpoint_every,
            "learning_rate": self.args.learning_rate,
            "batch_size": self.args.batch_size,
            "max_retries": self.args.max_retries,
            "expected_eval_episodes": self.args.expected_eval_episodes,
            "max_train_gpus": self.args.max_train_gpus,
            "retry_backoff": self.args.retry_backoff,
            "screen_epochs": list(SCREEN_EPOCHS),
            "candidates": list(SCREEN_CANDIDATES),
        }

    def _eval_protocol(self):
        if self.stage == "diagnose":
            return {
                "id_repeats": 1,
                "train_repeats": 1,
                "hard_repeats": 2,
                "id_seed_count": 20,
                "train_seed_count": 20,
                "hard_seed_count": 20,
                "include_hard": True,
                "expected_episodes": 80,
            }
        if self.stage == "screen":
            return {
                "id_repeats": 1,
                "train_repeats": 1,
                "hard_repeats": 0,
                "id_seed_count": 20,
                "train_seed_count": 10,
                "hard_seed_count": 0,
                "include_hard": False,
                "expected_episodes": 30,
            }
        return {
            "id_repeats": 3,
            "train_repeats": 3,
            "hard_repeats": 8,
            "id_seed_count": None,
            "train_seed_count": None,
            "hard_seed_count": None,
            "include_hard": True,
            "expected_episodes": 760,
        }

    def _shard_result_is_complete(self, path):
        if not Path(path).is_file():
            return False
        try:
            payload = read_json(path)
            progress = payload.get("progress", {})
            return bool(progress.get("complete")) and int(progress.get("completed_episodes", -1)) == len(
                payload.get("rows", [])
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    def _refresh_diagnose_job_commands(self, state):
        """Rebuild eval commands/artifacts so resumed runs use the current output layout."""
        output_dir = PHASE2B / "diagnose" / "eval"
        protocol = self._eval_protocol()
        for job in state["jobs"].values():
            if job.get("kind") != "eval":
                continue
            task = job["task"]
            variant = job["variant"]
            shard = int(job["shard"])
            ckpt = diagnose_ckpt_path(task, variant)
            job["command"] = self._eval_command(task, variant, ckpt, shard, output_dir)
            job["artifact"] = str(
                output_dir / task / f"{variant}_shard_{shard:02d}_of_{self.eval_shards:02d}.json"
            )
            job["expected_episodes"] = protocol["expected_episodes"]
            job["num_shards"] = self.eval_shards
            group = state["groups"][job["group"]]
            group["output_dir"] = str(output_dir)
            group["artifact"] = str(output_dir / task / f"{variant}.json")
            group["checkpoint"] = str(ckpt)
            group["expected_episodes"] = protocol["expected_episodes"]

    def _load_or_create_state(self):
        if self.state_path.exists():
            state = read_json(self.state_path)
            if state.get("config") != self.config:
                raise RuntimeError(
                    f"State config differs from this run: {self.state_path}. "
                    "Use the original configuration or a different PHASE2B_STATE_PATH."
                )
            if self.stage == "diagnose":
                self._refresh_diagnose_job_commands(state)
            if self.args.retry_failed:
                for job in state["jobs"].values():
                    if job["status"] == "failed":
                        job["status"] = "pending"
                        job["attempts"] = 0
            for job in state["jobs"].values():
                if job["status"] == "running":
                    job["status"] = "pending"
                    job["pid"] = None
                    job["gpu"] = None
                    job["attempts"] = max(0, int(job.get("attempts", 0)) - 1)
            state["status"] = "running"
            state["resumed_at"] = now()
            return state

        jobs = {}
        groups = {}
        promotions = {}
        if self.stage == "diagnose":
            self._add_diagnose_jobs(jobs, groups)
        elif self.stage == "screen":
            self._add_screen_jobs(jobs, groups, promotions)
        elif self.stage == "full_seed0":
            self._add_full_seed0_jobs(jobs, groups, promotions)
        elif self.stage == "confirm_seeds":
            self._add_confirm_jobs(jobs, groups)
        else:
            raise ValueError(f"Unsupported stage: {self.stage}")

        return {
            "version": 1,
            "stage": self.stage,
            "status": "running",
            "created_at": now(),
            "updated_at": now(),
            "config": self.config,
            "jobs": jobs,
            "groups": groups,
            "promotions": promotions,
            "events": [],
            "gpu_assignments": {},
        }

    def _eval_command(self, task, variant, ckpt_path, shard, output_dir):
        protocol = self._eval_protocol()
        command = [
            "python",
            str(PHASE1 / "eval_per_seed.py"),
            "--task",
            task,
            "--task-config",
            "demo_clean",
            "--variant",
            variant,
            "--output-dir",
            str(output_dir),
            "--shard-id",
            str(shard),
            "--num-shards",
            str(self.eval_shards),
            "--resume",
            "--ckpt-path",
            str(ckpt_path),
            "--rollout-dir",
            str(PHASE1 / "rollouts_200"),
            "--hard-seeds-file",
            str(hard_seeds_file(task)),
            "--policy-seed-offset",
            "1000",
            "--id-repeats",
            str(protocol["id_repeats"]),
            "--train-repeats",
            str(protocol["train_repeats"]),
            "--hard-repeats",
            str(protocol["hard_repeats"]),
        ]
        if protocol["id_seed_count"] is not None:
            command.extend(["--id-seed-count", str(protocol["id_seed_count"])])
        if protocol["train_seed_count"] is not None:
            command.extend(["--train-seed-count", str(protocol["train_seed_count"])])
        if protocol["hard_seed_count"] is not None:
            command.extend(["--hard-seed-count", str(protocol["hard_seed_count"])])
        if not protocol["include_hard"]:
            command.append("--no-include-hard")
        return command

    def _add_diagnose_jobs(self, jobs, groups):
        prep_ckpt = "prep:build_normalizer_ckpts"
        jobs[prep_ckpt] = {
            "id": prep_ckpt,
            "kind": "prep",
            "status": "pending",
            "artifact": str(PHASE2B / "diagnose" / "normalizer_ckpts.json"),
            "log": str(self.log_dir / "prep_build_normalizer_ckpts.log"),
            "command": ["python", str(PHASE2B / "build_normalizer_ckpts.py")],
        }
        prep_stats = "prep:compare_source_stats"
        jobs[prep_stats] = {
            "id": prep_stats,
            "kind": "prep",
            "status": "pending",
            "artifact": str(PHASE2B / "diagnose" / "source_stats.json"),
            "log": str(self.log_dir / "prep_compare_source_stats.log"),
            "command": ["python", str(PHASE2B / "compare_source_stats.py")],
        }
        protocol = self._eval_protocol()
        for task in self.args.tasks:
            for variant in DIAGNOSE_VARIANTS:
                ckpt = diagnose_ckpt_path(task, variant)
                group_id = f"eval:{task}:{variant}"
                output_dir = PHASE2B / "diagnose" / "eval"
                groups[group_id] = {
                    "id": group_id,
                    "task": task,
                    "variant": variant,
                    "dependency": prep_ckpt,
                    "status": "pending",
                    "attempts": 0,
                    "output_dir": str(output_dir),
                    "artifact": str(output_dir / task / f"{variant}.json"),
                    "checkpoint": str(ckpt),
                    "expected_episodes": protocol["expected_episodes"],
                }
                for shard in range(self.eval_shards):
                    job_id = f"{group_id}:shard{shard:02d}"
                    jobs[job_id] = {
                        "id": job_id,
                        "kind": "eval",
                        "task": task,
                        "variant": variant,
                        "shard": shard,
                        "num_shards": self.eval_shards,
                        "dependency": prep_ckpt,
                        "group": group_id,
                        "status": "pending",
                        "gpu": None,
                        "pid": None,
                        "attempts": 0,
                        "artifact": str(
                            output_dir / task / f"{variant}_shard_{shard:02d}_of_{self.eval_shards:02d}.json"
                        ),
                        "log": str(self.log_dir / f"eval_{task}_{variant}_shard{shard:02d}.log"),
                        "command": self._eval_command(task, variant, ckpt, shard, output_dir),
                        "expected_episodes": protocol["expected_episodes"],
                    }
        gate_id = "gate:diagnose_summary"
        jobs[gate_id] = {
            "id": gate_id,
            "kind": "gate",
            "status": "pending",
            "artifact": str(PHASE2B / "diagnose" / "diagnose_summary.json"),
            "depends_on_groups": [f"eval:{task}:{variant}" for task in self.args.tasks for variant in DIAGNOSE_VARIANTS],
        }

    def _train_command(self, task, candidate, seed, target_epoch):
        cfg = CANDIDATE_CONFIGS[candidate]
        expert_steps = compute_expert_steps(task, self.args.batch_size)
        return [
            "bash",
            str(PHASE2B / "finetune.sh"),
            task,
            candidate,
            "{gpu}",
            str(seed),
            str(target_epoch),
            "14",
            str(expert_steps),
            str(self.args.learning_rate),
            cfg["expert_ratio"],
            cfg["dataset_suffix"],
            str(self.args.checkpoint_every),
            cfg["loss_mode"],
            cfg["lambda_expert"],
            cfg["lambda_rollout"],
            "checkpoint",
        ]

    def _add_screen_jobs(self, jobs, groups, promotions):
        protocol = self._eval_protocol()
        for seed in self.args.train_seeds:
            for task in self.args.tasks:
                for candidate in SCREEN_CANDIDATES:
                    prev_train = None
                    prev_eval_group = None
                    for epoch in SCREEN_EPOCHS:
                        train_id = f"train:{task}:{candidate}:seed{seed}:to{epoch}"
                        ckpt = phase2b_checkpoint_dir(task, candidate, seed) / f"{epoch}.ckpt"
                        jobs[train_id] = {
                            "id": train_id,
                            "kind": "train",
                            "task": task,
                            "candidate": candidate,
                            "train_seed": seed,
                            "target_epoch": epoch,
                            "status": "pending",
                            "blocked_by": prev_eval_group,
                            "gpu": None,
                            "pid": None,
                            "attempts": 0,
                            "artifact": str(ckpt),
                            "log": str(self.log_dir / f"finetune_{task}_{candidate}_seed{seed}_to{epoch}.log"),
                            "command": self._train_command(task, candidate, seed, epoch),
                        }
                        group_id = f"screen:{task}:{candidate}:seed{seed}:epoch{epoch}"
                        output_dir = self.eval_dir / f"train_seed_{seed}" / "screen"
                        groups[group_id] = {
                            "id": group_id,
                            "task": task,
                            "variant": f"{candidate}_epoch{epoch}",
                            "candidate": candidate,
                            "train_seed": seed,
                            "epoch": epoch,
                            "dependency": train_id,
                            "status": "pending",
                            "attempts": 0,
                            "output_dir": str(output_dir),
                            "artifact": str(output_dir / task / f"{candidate}_epoch{epoch}.json"),
                            "checkpoint": str(ckpt),
                            "gate": "screen",
                            "expected_episodes": protocol["expected_episodes"],
                        }
                        for shard in range(self.eval_shards):
                            job_id = f"{group_id}:shard{shard:02d}"
                            jobs[job_id] = {
                                "id": job_id,
                                "kind": "eval",
                                "task": task,
                                "variant": f"{candidate}_epoch{epoch}",
                                "shard": shard,
                                "num_shards": self.eval_shards,
                                "dependency": train_id,
                                "group": group_id,
                                "status": "pending",
                                "gpu": None,
                                "pid": None,
                                "attempts": 0,
                                "artifact": str(
                                    output_dir
                                    / task
                                    / f"{candidate}_epoch{epoch}_shard_{shard:02d}_of_{self.eval_shards:02d}.json"
                                ),
                                "log": str(
                                    self.log_dir
                                    / f"eval_{task}_{candidate}_seed{seed}_epoch{epoch}_shard{shard:02d}.log"
                                ),
                                "command": self._eval_command(
                                    task,
                                    f"{candidate}_epoch{epoch}",
                                    ckpt,
                                    shard,
                                    output_dir,
                                ),
                                "expected_episodes": protocol["expected_episodes"],
                            }
                        gate_id = f"gate:{group_id}"
                        jobs[gate_id] = {
                            "id": gate_id,
                            "kind": "gate",
                            "status": "pending",
                            "artifact": str(PHASE2B / "promotions" / f"{group_id}.json"),
                            "depends_on_group": group_id,
                            "gate": "screen",
                            "task": task,
                            "candidate": candidate,
                            "epoch": epoch,
                            "previous_group": prev_eval_group,
                        }
                        promotions[group_id] = {
                            "status": "pending",
                            "task": task,
                            "candidate": candidate,
                            "epoch": epoch,
                            "train_seed": seed,
                        }
                        prev_train = train_id
                        prev_eval_group = group_id

    def _load_screen_winners(self):
        screen_state = self.args.screen_state_path
        if not screen_state.exists():
            raise FileNotFoundError(
                f"Missing screen state for full_seed0: {screen_state}. Run screen stage first."
            )
        screen_payload = read_json(screen_state)
        winners = []
        for group_id, promo in screen_payload.get("promotions", {}).items():
            if promo.get("status") == "promoted":
                winners.append(
                    (
                        promo["metrics"]["id_mean_sr"] + 0.5 * promo["metrics"]["train_mean_sr"],
                        promo["task"],
                        promo["candidate"],
                        int(promo["epoch"]),
                    )
                )
        winners.sort(reverse=True)
        return winners[: self.args.full_top_k]

    def _add_full_seed0_jobs(self, jobs, groups, promotions):
        protocol = self._eval_protocol()
        winners = self._load_screen_winners()
        if not winners:
            raise RuntimeError("No screen winners found for full_seed0 stage.")
        for seed in self.args.train_seeds:
            for _, task, candidate, epoch in winners:
                ckpt = phase2b_checkpoint_dir(task, candidate, seed) / f"{epoch}.ckpt"
                group_id = f"full:{task}:{candidate}:seed{seed}:epoch{epoch}"
                output_dir = self.eval_dir / f"train_seed_{seed}"
                groups[group_id] = {
                    "id": group_id,
                    "task": task,
                    "variant": candidate,
                    "candidate": candidate,
                    "train_seed": seed,
                    "epoch": epoch,
                    "status": "pending",
                    "attempts": 0,
                    "output_dir": str(output_dir),
                    "artifact": str(output_dir / task / f"{candidate}.json"),
                    "checkpoint": str(ckpt),
                    "gate": "full",
                    "expected_episodes": protocol["expected_episodes"],
                }
                for shard in range(self.eval_shards):
                    job_id = f"{group_id}:shard{shard:02d}"
                    jobs[job_id] = {
                        "id": job_id,
                        "kind": "eval",
                        "task": task,
                        "variant": candidate,
                        "shard": shard,
                        "num_shards": self.eval_shards,
                        "group": group_id,
                        "status": "pending",
                        "gpu": None,
                        "pid": None,
                        "attempts": 0,
                        "artifact": str(
                            output_dir
                            / task
                            / f"{candidate}_shard_{shard:02d}_of_{self.eval_shards:02d}.json"
                        ),
                        "log": str(self.log_dir / f"eval_full_{task}_{candidate}_seed{seed}_shard{shard:02d}.log"),
                        "command": self._eval_command(task, candidate, ckpt, shard, output_dir),
                        "expected_episodes": protocol["expected_episodes"],
                    }
                gate_id = f"gate:{group_id}"
                jobs[gate_id] = {
                    "id": gate_id,
                    "kind": "gate",
                    "status": "pending",
                    "artifact": str(PHASE2B / "promotions" / f"{group_id}.json"),
                    "depends_on_group": group_id,
                    "gate": "full",
                    "task": task,
                    "candidate": candidate,
                    "epoch": epoch,
                }
                promotions[group_id] = {"status": "pending", "task": task, "candidate": candidate, "epoch": epoch}

    def _load_full_winner(self):
        full_state = self.args.full_state_path
        if not full_state.exists():
            raise FileNotFoundError(f"Missing full_seed0 state: {full_state}")
        payload = read_json(full_state)
        winners = [
            promo
            for promo in payload.get("promotions", {}).values()
            if promo.get("status") == "promoted"
        ]
        if len(winners) != 1:
            raise RuntimeError(
                f"confirm_seeds requires exactly one full_seed0 winner, found {len(winners)}"
            )
        return winners[0]

    def _add_confirm_jobs(self, jobs, groups):
        winner = self._load_full_winner()
        task = winner["task"]
        candidate = winner["candidate"]
        epoch = int(winner["epoch"])
        protocol = self._eval_protocol()
        for seed in self.args.confirm_seeds:
            train_id = f"train:{task}:{candidate}:seed{seed}:to{epoch}"
            ckpt = phase2b_checkpoint_dir(task, candidate, seed) / f"{epoch}.ckpt"
            jobs[train_id] = {
                "id": train_id,
                "kind": "train",
                "task": task,
                "candidate": candidate,
                "train_seed": seed,
                "target_epoch": epoch,
                "status": "pending",
                "gpu": None,
                "pid": None,
                "attempts": 0,
                "artifact": str(ckpt),
                "log": str(self.log_dir / f"finetune_{task}_{candidate}_seed{seed}_to{epoch}.log"),
                "command": self._train_command(task, candidate, seed, epoch),
            }
            group_id = f"confirm:{task}:{candidate}:seed{seed}:epoch{epoch}"
            output_dir = self.eval_dir / f"train_seed_{seed}"
            groups[group_id] = {
                "id": group_id,
                "task": task,
                "variant": candidate,
                "dependency": train_id,
                "status": "pending",
                "attempts": 0,
                "output_dir": str(output_dir),
                "artifact": str(output_dir / task / f"{candidate}.json"),
                "checkpoint": str(ckpt),
                "expected_episodes": protocol["expected_episodes"],
            }
            for shard in range(self.eval_shards):
                job_id = f"{group_id}:shard{shard:02d}"
                jobs[job_id] = {
                    "id": job_id,
                    "kind": "eval",
                    "task": task,
                    "variant": candidate,
                    "shard": shard,
                    "num_shards": self.eval_shards,
                    "dependency": train_id,
                    "group": group_id,
                    "status": "pending",
                    "gpu": None,
                    "pid": None,
                    "attempts": 0,
                    "artifact": str(
                        output_dir / task / f"{candidate}_shard_{shard:02d}_of_{self.eval_shards:02d}.json"
                    ),
                    "log": str(self.log_dir / f"eval_confirm_{task}_{candidate}_seed{seed}_shard{shard:02d}.log"),
                    "command": self._eval_command(task, candidate, ckpt, shard, output_dir),
                    "expected_episodes": protocol["expected_episodes"],
                }

    def _event(self, message):
        self.state["events"].append({"time": now(), "message": message})
        self.state["events"] = self.state["events"][-200:]
        print(f"[{now()}] {message}", flush=True)

    def _legacy_diagnose_eval_path(self, artifact_path):
        path = Path(artifact_path)
        legacy = path.parent / path.parent.name / path.name
        if legacy == path:
            return None
        return legacy

    def _migrate_legacy_eval_artifacts(self):
        if self.stage != "diagnose":
            return
        for job in self.state["jobs"].values():
            if job["kind"] != "eval":
                continue
            target = Path(job["artifact"])
            if self._shard_result_is_complete(target):
                continue
            legacy = self._legacy_diagnose_eval_path(target)
            if legacy is None or not legacy.is_file():
                continue
            if not self._shard_result_is_complete(legacy):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy, target)
            self._event(f"Migrated legacy eval artifact {legacy} -> {target}")

    def _refresh_from_artifacts(self):
        self._migrate_legacy_eval_artifacts()
        for group in self.state["groups"].values():
            expected = group.get("expected_episodes", self.args.expected_eval_episodes)
            merged = Path(group["artifact"])
            if merged_result_is_complete(
                merged,
                group["task"],
                group.get("variant"),
                Path(group["checkpoint"]),
                expected,
            ):
                group["status"] = "completed"
                for job in self.state["jobs"].values():
                    if job.get("group") == group["id"]:
                        job["status"] = "completed"
                        job["pid"] = None
                        job["gpu"] = None
        for job in self.state["jobs"].values():
            if job["id"] not in self.processes and job["status"] != "completed" and artifact_is_complete(job):
                job["status"] = "completed"
                job["pid"] = None
                job["gpu"] = None

    def _gpu_assignments(self):
        result = {str(gpu): {"mode": "idle", "train": None, "eval": []} for gpu in self.gpus}
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
                    "target_epoch": job.get("target_epoch"),
                }
            elif job["kind"] == "eval":
                completed_episodes = 0
                path = Path(job["artifact"])
                if path.is_file():
                    try:
                        completed_episodes = len(read_json(path).get("rows", []))
                    except (OSError, ValueError, json.JSONDecodeError):
                        pass
                shard = int(job["shard"])
                expected = job.get("expected_episodes", self.args.expected_eval_episodes)
                target = max(0, (expected + self.eval_shards - 1 - shard) // self.eval_shards)
                job["progress"] = {"completed_episodes": completed_episodes, "total_episodes": target}

    def _start_job(self, job, gpu):
        if job["kind"] == "prep":
            log_path = Path(job["log"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log:
                result = subprocess.run(job["command"], cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode == 0 and artifact_is_complete(job):
                job["status"] = "completed"
                self._event(f"Completed {job['id']}")
            else:
                job["status"] = "failed"
                job["message"] = f"exit={result.returncode}"
                self._event(f"FAILED {job['id']} (exit={result.returncode}); see {log_path}")
            return

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
        self.processes[job["id"]] = {"process": proc, "log_file": log_file, "gpu": gpu, "job": job}
        self._event(f"Started {job['id']} on GPU {gpu}, PID {proc.pid}")

    def _finish_process(self, job_id, returncode):
        info = self.processes.pop(job_id)
        info["log_file"].close()
        job = info["job"]
        job["pid"] = None
        job["gpu"] = None
        if returncode == 0 and artifact_is_complete(job):
            job["status"] = "completed"
            self._event(f"Completed {job_id}")
        elif self.stopping:
            job["status"] = "pending"
            job["attempts"] = max(0, int(job["attempts"]) - 1)
        elif job["attempts"] < self.args.max_retries:
            job["status"] = "pending"
            job["not_before"] = time.time() + self.args.retry_backoff * job["attempts"]
            self._event(f"Retrying {job_id} after exit {returncode}")
        else:
            job["status"] = "failed"
            self._event(f"FAILED {job_id}")

    def _poll(self):
        for job_id, info in list(self.processes.items()):
            rc = info["process"].poll()
            if rc is not None:
                self._finish_process(job_id, rc)

    def _dependency_complete(self, job):
        dependency = job.get("dependency")
        if dependency and self.state["jobs"][dependency]["status"] != "completed":
            return False
        blocked_by = job.get("blocked_by")
        if blocked_by:
            promo = self.state["promotions"].get(blocked_by, {})
            if promo.get("status") != "promoted":
                return False
        return True

    def _pending_jobs(self, kind):
        return [
            job
            for job in self.state["jobs"].values()
            if job["kind"] == kind
            and job["status"] == "pending"
            and self._dependency_complete(job)
            and float(job.get("not_before", 0)) <= time.time()
        ]

    def _schedule(self):
        pending_train = self._pending_jobs("train")
        pending_eval = [
            job
            for job in self._pending_jobs("eval")
            if self.state["groups"][job["group"]]["status"] != "completed"
        ]
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
            shard_jobs = [job for job in self.state["jobs"].values() if job.get("group") == group["id"]]
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
            if "_epoch" in group["variant"]:
                command[command.index("--variant") + 1] = group["variant"]
            log_path = self.log_dir / f"merge_{group['task']}_{group['variant']}.log"
            with log_path.open("a", encoding="utf-8") as log:
                subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
            expected = group.get("expected_episodes", self.args.expected_eval_episodes)
            if merged_result_is_complete(
                Path(group["artifact"]),
                group["task"],
                group["variant"],
                Path(group["checkpoint"]),
                expected,
            ):
                group["status"] = "completed"
                group["completed_at"] = now()
                self._event(f"Merged {group['id']}")

    def _run_gate(self, job):
        if job["kind"] != "gate" or job["status"] == "completed":
            return
        if job["id"] == "gate:diagnose_summary":
            summary = {"tasks": {}, "created_at": now()}
            for task in self.args.tasks:
                a0 = read_json(PHASE2B / "diagnose" / "eval" / task / "A0_base.json")
                a2 = read_json(PHASE2B / "diagnose" / "eval" / task / "A2_mixed_norm.json")
                summary["tasks"][task] = evaluate_diagnose(task, a0, a2)
            atomic_write_json(job["artifact"], summary)
            job["status"] = "completed"
            self._event("Completed diagnose summary gate")
            return

        group_id = job["depends_on_group"]
        if self.state["groups"][group_id]["status"] != "completed":
            return
        eval_payload = read_json(self.state["groups"][group_id]["artifact"])
        base_metrics = load_base_metrics(job["task"], base_eval_path(job["task"]))
        previous_payload = None
        if job.get("previous_group"):
            prev_group = job["previous_group"]
            prev_path = self.state["groups"][prev_group]["artifact"]
            if Path(prev_path).is_file():
                previous_payload = read_json(prev_path)
        if job["gate"] == "screen":
            result = evaluate_screen(job["task"], eval_payload, base_metrics, previous_payload)
            result["status"] = "promoted" if result["passed"] else "eliminated"
            result["group_id"] = group_id
            result["candidate"] = job["candidate"]
            result["epoch"] = job["epoch"]
            atomic_write_json(job["artifact"], result)
            self.state["promotions"][group_id] = result
            job["status"] = "completed"
            self._event(
                f"Screen gate {group_id}: {'promoted' if result['passed'] else 'eliminated'} "
                f"({'; '.join(result['reasons']) or 'ok'})"
            )
        elif job["gate"] == "full":
            result = evaluate_full(job["task"], eval_payload, base_metrics)
            result["status"] = "promoted" if result["passed"] else "eliminated"
            result["group_id"] = group_id
            atomic_write_json(job["artifact"], result)
            self.state["promotions"][group_id] = result
            job["status"] = "completed"
            self._event(
                f"Full gate {group_id}: {'promoted' if result['passed'] else 'eliminated'} "
                f"({'; '.join(result['reasons']) or 'ok'})"
            )

    def _process_gates(self):
        for job in self.state["jobs"].values():
            if job["kind"] != "gate" or job["status"] == "completed":
                continue
            if job["id"] == "gate:diagnose_summary":
                if all(
                    self.state["groups"][group_id]["status"] == "completed"
                    for group_id in job["depends_on_groups"]
                ):
                    self._run_gate(job)
                continue
            group_id = job.get("depends_on_group")
            if group_id and self.state["groups"][group_id]["status"] == "completed":
                self._run_gate(job)

    def _aggregate(self):
        if self.stage == "diagnose":
            return
        command = [
            "python",
            str(PHASE2B / "aggregate_eval.py"),
            "--eval-dir",
            str(self.eval_dir),
            "--train-seeds",
            *[str(seed) for seed in self.args.train_seeds],
            "--tasks",
            *self.args.tasks,
            "--output",
            str(self.eval_dir / f"summary_{self.stage}.json"),
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    def _all_complete(self):
        return all(job["status"] == "completed" for job in self.state["jobs"].values()) and all(
            group["status"] == "completed" for group in self.state["groups"].values()
        )

    def stop(self, final_status="interrupted"):
        if self.stopping:
            return
        self.stopping = True
        for info in self.processes.values():
            try:
                os.killpg(info["process"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        while self.processes:
            self._poll()
            time.sleep(0.1)
        self.state["status"] = final_status
        self._save_state()

    def run(self):
        for job in [job for job in self.state["jobs"].values() if job["kind"] == "prep" and job["status"] == "pending"]:
            self._start_job(job, gpu=None)
        while not self.stopping:
            self._poll()
            self._refresh_from_artifacts()
            self._merge_ready_groups()
            self._process_gates()
            if any(job["status"] == "failed" for job in self.state["jobs"].values()):
                self.state["status"] = "failed"
                self._save_state()
                raise RuntimeError(f"Phase 2b has terminal failures; inspect {self.state_path}")
            if self._all_complete():
                self._aggregate()
                self.state["status"] = "completed"
                self.state["completed_at"] = now()
                self._event(f"Phase 2b stage {self.stage} completed")
                self._save_state()
                return
            self._schedule()
            self._save_state()
            time.sleep(self.args.poll_interval)


def parse_args():
    parser = argparse.ArgumentParser(description="Run resumable Phase 2b staged experiments.")
    parser.add_argument("stage", choices=("diagnose", "screen", "full_seed0", "confirm_seeds"))
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--confirm-seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--gpus", nargs="+", type=int, default=list(DEFAULT_GPUS))
    parser.add_argument("--eval-per-gpu", type=int, default=3)
    parser.add_argument("--eval-shards", type=int)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=60.0)
    parser.add_argument("--max-train-gpus", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--expected-eval-episodes", type=int, default=760)
    parser.add_argument("--full-top-k", type=int, default=2)
    parser.add_argument("--state-path", type=Path, default=PHASE2B / "run_state.json")
    parser.add_argument("--screen-state-path", type=Path, default=PHASE2B / "run_state_screen.json")
    parser.add_argument("--full-state-path", type=Path, default=PHASE2B / "run_state_full_seed0.json")
    args = parser.parse_args()
    if args.max_train_gpus is None:
        args.max_train_gpus = len(args.gpus)
    if args.stage == "full_seed0":
        args.expected_eval_episodes = 760
    elif args.stage == "screen":
        args.expected_eval_episodes = 30
    elif args.stage == "diagnose":
        args.expected_eval_episodes = 80
    return args


def main():
    args = parse_args()
    args.state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.state_path.with_suffix(args.state_path.suffix + ".lock")
    lock_file = lock_path.open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"Another Phase 2b scheduler owns {lock_path}")

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
