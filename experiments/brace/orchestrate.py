#!/usr/bin/env python3
"""Resumable BRACE developmental screen trainer/evaluator."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
PHASE1_DIR = REPO_ROOT / "experiments" / "phase1"
DP_DIR = REPO_ROOT / "policy" / "DP"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.replay_audit import git_commit, read_json, write_json_atomic

TRAIN_METHODS = ("N1", "B1", "B2", "B3")
METHOD_DATASET = {"N1": "N1", "U1": "N1", "B1": "B1", "B2": "N1", "B3": "B1"}


def utc_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def base_checkpoint(task: str) -> Path:
    return DP_DIR / "checkpoints" / f"{task}-demo_clean-200-0" / "600.ckpt"


def expert_dataset(task: str) -> Path:
    return DP_DIR / "data_phase1_200" / f"{task}-expert_only.zarr"


def hard_seeds(task: str) -> Path:
    return PHASE1_DIR / "eval_results_200" / "hard_eval_seeds" / f"{task}.json"


def screen_manifest(run_label: str, dataset: str) -> Path:
    return BRACE_DIR / "datasets" / f"{run_label}_{dataset}.jsonl"


def resolve_anchor_summary(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    pointer = BRACE_DIR / "runs" / "LATEST_anchor_smoke"
    if pointer.is_file():
        candidate = Path(pointer.read_text(encoding="utf-8").strip()) / "summary.json"
        if candidate.is_file():
            return candidate
    return BRACE_DIR / "anchor_smoke" / "summary.json"


def checkpoint_path(task: str, checkpoint_label: str, method: str, seed: int, epoch: int) -> Path:
    return DP_DIR / "checkpoints" / f"{task}-brace-{checkpoint_label}-{method}-{seed}" / f"{epoch}.ckpt"


def artifact_complete(job: dict[str, Any]) -> bool:
    artifact = Path(job["artifact"])
    if job["kind"] == "prepare":
        return artifact.is_file()
    if job["kind"] == "train":
        return artifact.is_file() and artifact.stat().st_size > 0 and Path(str(artifact) + ".complete").is_file()
    if job["kind"] == "eval":
        if not artifact.is_file():
            return False
        try:
            payload = read_json(artifact)
            progress = payload["progress"]
            return bool(progress["complete"]) and int(progress["completed_episodes"]) == 60
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
    return False


def compute_steps_per_epoch(path: Path, batch_size: int) -> int:
    import zarr

    root = zarr.open(str(path), mode="r")
    frames = int(root["meta/episode_ends"][-1])
    steps = frames // batch_size
    if steps < 1:
        raise RuntimeError(f"expert dataset is too small for batch_size={batch_size}: {path}")
    return steps


def eval_command(task: str, variant: str, ckpt: Path, output_dir: Path) -> list[str]:
    return [
        "python",
        str(PHASE1_DIR / "eval_per_seed.py"),
        "--task",
        task,
        "--task-config",
        "demo_clean",
        "--variant",
        variant,
        "--ckpt-path",
        str(ckpt),
        "--output-dir",
        str(output_dir),
        "--hard-seeds-file",
        str(hard_seeds(task)),
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


def create_state(args, protocol: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    task = args.task
    run_label = args.run_label
    train_cfg = protocol["training"]
    checkpoint_label = run_dir.name.split("_screen_", 1)[0]
    jobs: dict[str, dict[str, Any]] = {}
    datasets: dict[str, str] = {}
    missing_inputs: list[str] = []
    expert = expert_dataset(task)
    if not expert.is_dir():
        missing_inputs.append(str(expert))
        steps_per_epoch = None
    else:
        steps_per_epoch = compute_steps_per_epoch(expert, int(train_cfg["batch_size"]))

    for dataset in ("B1", "N1"):
        manifest = screen_manifest(run_label, dataset)
        output = run_dir / "datasets" / f"{task}_{dataset}.zarr"
        datasets[dataset] = str(output)
        job_id = f"prepare:{dataset}"
        jobs[job_id] = {
            "id": job_id,
            "kind": "prepare",
            "status": "pending",
            "dependency": None,
            "artifact": str(output / "brace_dataset_manifest.json"),
            "log": str(run_dir / "logs" / f"{job_id.replace(':', '_')}.log"),
            "command": [
                "python",
                str(BRACE_DIR / "build_screen_dataset.py"),
                "--expert-zarr",
                str(expert),
                "--manifest",
                str(manifest),
                "--output",
                str(output),
                "--traced-root",
                str(args.traced_rollout_dir),
            ],
            "attempts": 0,
        }
        if not manifest.is_file():
            missing_inputs.append(str(manifest))

    eval_dir = run_dir / "eval"
    base_id = "eval:base"
    base_artifact = eval_dir / task / "base.json"
    jobs[base_id] = {
        "id": base_id,
        "kind": "eval",
        "method": "base",
        "epoch": 0,
        "status": "pending",
        "dependency": None,
        "artifact": str(base_artifact),
        "log": str(run_dir / "logs" / "eval_base.log"),
        "command": eval_command(task, "base", base_checkpoint(task), eval_dir),
        "attempts": 0,
    }
    if not base_checkpoint(task).is_file():
        missing_inputs.append(str(base_checkpoint(task)))
    if not hard_seeds(task).is_file():
        missing_inputs.append(str(hard_seeds(task)))
    anchor_summary = resolve_anchor_summary(args.anchor_summary)
    if not anchor_summary.is_file():
        missing_inputs.append(str(anchor_summary))
    else:
        anchor_payload = read_json(anchor_summary)
        if not anchor_payload.get("passed") or anchor_payload.get("protocol_revision") != protocol["protocol_revision"]:
            missing_inputs.append(
                f"passing {protocol['protocol_revision']} anchor smoke:{anchor_summary}"
            )

    for method in TRAIN_METHODS:
        dependency = f"prepare:{METHOD_DATASET[method]}"
        for epoch in protocol["screen_epochs"]:
            train_id = f"train:{method}:epoch{epoch}"
            ckpt = checkpoint_path(task, checkpoint_label, method, int(protocol["train_seed"]), int(epoch))
            jobs[train_id] = {
                "id": train_id,
                "kind": "train",
                "method": method,
                "epoch": int(epoch),
                "status": "pending",
                "dependency": dependency,
                "artifact": str(ckpt),
                "log": str(run_dir / "logs" / f"{train_id.replace(':', '_')}.log"),
                "command": [
                    "bash",
                    str(BRACE_DIR / "finetune.sh"),
                    task,
                    method,
                    "{gpu}",
                    str(protocol["train_seed"]),
                    str(epoch),
                    datasets[METHOD_DATASET[method]],
                    checkpoint_label,
                    str(steps_per_epoch or 0),
                    str(train_cfg["learning_rate"]),
                    str(train_cfg["rollout_per_batch"]),
                    str(train_cfg["batch_size"]),
                ],
                "attempts": 0,
            }
            eval_id = f"eval:{method}:epoch{epoch}"
            artifact = eval_dir / task / f"{method}_epoch{epoch}.json"
            jobs[eval_id] = {
                "id": eval_id,
                "kind": "eval",
                "method": method,
                "epoch": int(epoch),
                "status": "pending",
                "dependency": train_id,
                "artifact": str(artifact),
                "log": str(run_dir / "logs" / f"{eval_id.replace(':', '_')}.log"),
                "command": eval_command(task, f"{method}_epoch{epoch}", ckpt, eval_dir),
                "attempts": 0,
            }
            dependency = eval_id

    return {
        "schema_version": 2,
        "stage": "development_screen",
        "status": "planned" if args.dry_run else "running",
        "task": task,
        "run_label": run_label,
        "run_dir": str(run_dir),
        "checkpoint_label": checkpoint_label,
        "protocol_path": str(args.protocol),
        "protocol_sha256": file_sha256(args.protocol),
        "anchor_summary": str(anchor_summary),
        "git_commit": git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": datasets,
        "steps_per_epoch": steps_per_epoch,
        "missing_inputs": sorted(set(missing_inputs)),
        "jobs": jobs,
        "events": [],
    }


def split_metrics(payload: dict[str, Any]) -> dict[str, float]:
    return {
        split: float(payload.get("splits", {}).get(split, {}).get("mean_sr", 0.0))
        for split in ("id_heldout", "train_seen", "hard_20")
    }


def summarize_screen(state: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    base = split_metrics(read_json(Path(state["jobs"]["eval:base"]["artifact"])))
    methods: dict[str, Any] = {}
    for method in TRAIN_METHODS:
        methods[method] = {}
        for epoch in protocol["screen_epochs"]:
            payload = read_json(Path(state["jobs"][f"eval:{method}:epoch{epoch}"]["artifact"]))
            metrics = split_metrics(payload)
            normalized = {
                split: metrics[split] / max(base[split], 0.05)
                for split in metrics
            }
            methods[method][str(epoch)] = {
                "metrics": metrics,
                "normalized": normalized,
                "id_train_gate": (
                    metrics["id_heldout"] >= protocol["promotion"]["id_fraction_of_base"] * base["id_heldout"]
                    and metrics["train_seen"] >= protocol["promotion"]["train_fraction_of_base"] * base["train_seen"]
                ),
                "selection_score": min(normalized.values()),
                "artifact": state["jobs"][f"eval:{method}:epoch{epoch}"]["artifact"],
            }
    methods["U1"] = methods["N1"]

    def best(method: str) -> dict[str, Any] | None:
        eligible = [
            dict(value, epoch=int(epoch))
            for epoch, value in methods[method].items()
            if value["id_train_gate"]
        ]
        return max(eligible, key=lambda row: (row["selection_score"], -row["epoch"])) if eligible else None

    bests = {method: best(method) for method in ("N1", "U1", "B1", "B2", "B3")}
    credit_passed = bool(
        bests["B1"] and bests["N1"] and bests["B1"]["selection_score"] > bests["N1"]["selection_score"]
    )
    preservation_passed = bool(
        bests["B2"] and bests["U1"] and bests["B2"]["selection_score"] > bests["U1"]["selection_score"]
    )
    integration_best = max(
        ((method, bests[method]) for method in ("B1", "B2", "B3") if bests[method]),
        key=lambda item: item[1]["selection_score"],
        default=None,
    )
    integration_passed = bool(credit_passed and preservation_passed and integration_best and integration_best[0] == "B3")
    return {
        "schema_version": 1,
        "complete": True,
        "passed": bool(credit_passed and preservation_passed and integration_passed),
        "developmental_only": True,
        "paper_claim_ready": False,
        "task": state["task"],
        "run_label": state["run_label"],
        "base": base,
        "methods": methods,
        "best_eligible": bests,
        "screens": {
            "credit_B1_vs_N1": {"passed": credit_passed},
            "preservation_U1_vs_B2": {"passed": preservation_passed, "U1_alias": "N1"},
            "integration_B1_B2_B3": {
                "passed": integration_passed,
                "best_method": integration_best[0] if integration_best else None,
                "requires_prior_screens": True,
            },
        },
        "limitations": [
            "11 B1 + 11 matched-random chunks are sufficient only for developmental falsification.",
            "U1 uses the N1 matched-random manifest so B2 and U1 see identical optimizer examples.",
            "No reserve split is inspected and no full/paper screen is promoted from this run.",
        ],
    }


class Runner:
    def __init__(self, args, state_path: Path, state: dict[str, Any], protocol: dict[str, Any]):
        self.args = args
        self.state_path = state_path
        self.state = state
        self.protocol = protocol
        self.running: dict[int, tuple[str, subprocess.Popen, Any]] = {}
        self.stopping = False

    def save(self):
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(self.state_path, self.state)

    def refresh(self):
        for job in self.state["jobs"].values():
            if artifact_complete(job):
                job["status"] = "completed"
            elif job["status"] == "running":
                job["status"] = "pending"

    def ready(self, job):
        dependency = job.get("dependency")
        return job["status"] == "pending" and (
            dependency is None or self.state["jobs"][dependency]["status"] == "completed"
        )

    def launch(self, gpu: int, job: dict[str, Any]):
        command = [str(gpu) if value == "{gpu}" else value for value in job["command"]]
        log_path = Path(job["log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a", encoding="utf-8")
        handle.write("COMMAND " + json.dumps(command) + "\n")
        handle.flush()
        process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)
        job.update(status="running", gpu=gpu, pid=process.pid, attempts=int(job["attempts"]) + 1)
        self.running[gpu] = (job["id"], process, handle)
        self.state["events"].append({"event": "launched", "job": job["id"], "gpu": gpu, "pid": process.pid})
        self.save()

    def poll(self):
        for gpu, (job_id, process, handle) in list(self.running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            job = self.state["jobs"][job_id]
            if code == 0 and artifact_complete(job):
                job["status"] = "completed"
            elif int(job["attempts"]) <= self.args.max_retries:
                job["status"] = "pending"
            else:
                job["status"] = "failed"
            job.update(exit_code=code, gpu=None, pid=None)
            self.state["events"].append({"event": job["status"], "job": job_id, "exit_code": code})
            del self.running[gpu]
            self.save()

    def stop(self, *_):
        self.stopping = True
        for _, process, _ in self.running.values():
            process.send_signal(signal.SIGINT)

    def run(self) -> int:
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        self.refresh()
        self.save()
        while not self.stopping:
            self.poll()
            failed = [job for job in self.state["jobs"].values() if job["status"] == "failed"]
            if failed:
                self.state["status"] = "failed"
                self.save()
                return 1
            if all(job["status"] == "completed" for job in self.state["jobs"].values()):
                summary = summarize_screen(self.state, self.protocol)
                summary_path = Path(self.state["run_dir"]) / "summary.json"
                write_json_atomic(summary_path, summary)
                self.state["status"] = "completed"
                self.state["summary"] = str(summary_path)
                self.save()
                from experiments.brace.stage_records import emit_stage_record

                emit_stage_record(
                    "development_screen",
                    summary=summary,
                    summary_path=summary_path,
                    artifact_dir=Path(self.state["run_dir"]),
                    run_dir=Path(self.state["run_dir"]),
                    tasks=[self.state["task"]],
                    label=self.state["run_label"],
                )
                return 0
            free_gpus = [gpu for gpu in self.args.gpus if gpu not in self.running]
            ready = [job for job in self.state["jobs"].values() if self.ready(job)]
            for gpu, job in zip(free_gpus, ready):
                self.launch(gpu, job)
            if not self.running and not ready:
                self.state["status"] = "blocked"
                self.save()
                return 2
            time.sleep(2)
        self.poll()
        self.state["status"] = "interrupted"
        self.save()
        return 130


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["screen", "full"])
    parser.add_argument("--protocol", type=Path, default=BRACE_DIR / "screen_protocol.v1.1.json")
    parser.add_argument("--task", default="place_container_plate")
    parser.add_argument("--run-label", default="place_pilot_v2.3")
    parser.add_argument("--traced-rollout-dir", type=Path, default=BRACE_DIR / "rollouts_traced_pilot")
    parser.add_argument("--anchor-summary", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--gpus", nargs="*", type=int, default=[0])
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.stage == "full":
        raise SystemExit("Full evaluation remains gated on a passing developmental screen and expanded confirm data.")
    protocol = read_json(args.protocol)

    if args.state:
        state_path = args.state
        state = read_json(state_path) if state_path.is_file() else None
        run_dir = state_path.parent
    else:
        state = None
        pointer = BRACE_DIR / "runs" / "LATEST_SCREEN_DEVELOPMENT"
        if not args.dry_run and pointer.is_file():
            candidate_dir = Path(pointer.read_text(encoding="utf-8").strip())
            candidate_state = candidate_dir / "state.json"
            if candidate_state.is_file():
                candidate = read_json(candidate_state)
                if (
                    candidate.get("task") == args.task
                    and candidate.get("run_label") == args.run_label
                    and candidate.get("status") in {"running", "interrupted", "blocked"}
                ):
                    run_dir = candidate_dir
                    state_path = candidate_state
                    state = candidate
        if state is None:
            run_dir = BRACE_DIR / "runs" / f"{utc_id()}_screen_{args.task}_development"
            state_path = run_dir / "state.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    if state is None:
        state = create_state(args, protocol, run_dir)
        meta = {
            "run_id": run_dir.name,
            "stage": "development_screen",
            "tasks": [args.task],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
        }
        write_json_atomic(run_dir / "meta.json", meta)
        write_json_atomic(state_path, state)
        pointer = BRACE_DIR / "runs" / "LATEST_SCREEN_DEVELOPMENT"
        pointer.write_text(str(run_dir) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps({
            "state": str(state_path),
            "job_count": len(state["jobs"]),
            "missing_inputs": state["missing_inputs"],
            "status": state["status"],
        }, indent=2))
        return 0
    if state["missing_inputs"]:
        raise SystemExit("Missing screen inputs:\n" + "\n".join(state["missing_inputs"]))
    return Runner(args, state_path, state, protocol).run()


if __name__ == "__main__":
    raise SystemExit(main())
