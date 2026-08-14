#!/usr/bin/env python3
"""Schedule place Base200 Line A *adaptation* eval (confirm_easy → confirm_hard).

Each eval GPU runs one logical job at a time with 3 concurrent shards via
run_eval_group.

Preservation is NOT auto-appended when a cohort later appears. Jobs and
variants are fixed when eval_state.json is first created (cohort_path is
usually null). After census freezes a cohort, launch the independent
preservation scheduler `_schedule_place_base200_v2_preservation_eval.py`
with a separate state file; do not rebuild this adaptation state in place.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

REPO = Path("/workspace/RoboTwin")
BRACE = REPO / "experiments" / "brace"
PHASE1 = REPO / "experiments" / "phase1"
DP = REPO / "policy" / "DP"
PY = "/root/miniconda/envs/RoboTwin/bin/python"
METHODS = ("U0", "N1", "B1", "B2", "B3")
SEEDS = (1, 2, 3, 4, 5)
CHECKPOINT_LABEL = "place_base200_v2"
EPOCHS = 10
WORKERS = 3
DEFAULT_EVAL_GPUS = (2, 3, 4, 5, 6, 7)


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cohort_path() -> Path | None:
    pointer = BRACE / "runs/LATEST_place_base200_line_a_preservation_cohort"
    if not pointer.is_file():
        return None
    path = Path(pointer.read_text(encoding="utf-8").strip())
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("frozen") or not payload.get("meets_min_untouched"):
        return None
    return path


def checkpoint_dir(method: str, seed: int) -> Path:
    name = f"place_container_plate-brace-{CHECKPOINT_LABEL}-{method}"
    return DP / "checkpoints" / f"{name}-{seed}"


def checkpoint_ready(method: str, seed: int, *, record_hash: bool = False) -> tuple[bool, dict[str, Any]]:
    cdir = checkpoint_dir(method, seed)
    ckpt = cdir / f"{EPOCHS}.ckpt"
    complete = cdir / f"{EPOCHS}.ckpt.complete"
    info: dict[str, Any] = {"checkpoint_dir": str(cdir)}
    if not ckpt.is_file() or ckpt.stat().st_size <= 0:
        return False, {**info, "reason": "missing_ckpt"}
    if not complete.is_file():
        return False, {**info, "reason": "missing_complete_marker"}
    if record_hash:
        info["checkpoint_sha256"] = file_sha256(ckpt)
    if method in ("B2", "B3"):
        feas = cdir / f"{EPOCHS}.feasibility.json"
        if not feas.is_file() or feas.stat().st_size <= 0:
            return False, {**info, "reason": "missing_feasibility"}
        if record_hash:
            info["feasibility_sha256"] = file_sha256(feas)
    return True, info


def build_adaptation_seeds(run_dir: Path) -> tuple[Path, Path]:
    manifest = json.loads((BRACE / "seeds/multitask_v1/place_container_plate.json").read_text(encoding="utf-8"))
    parts = manifest["partitions"]
    seeds_dir = run_dir / "eval_seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    easy = {
        "task": "place_container_plate",
        "eval_id": [int(x) for x in parts["confirm_easy"]],
        "train_rollout": [],
        "split": "confirm_easy",
        "task_config": "demo_clean",
    }
    hard = {
        "task": "place_container_plate",
        "eval_id": [int(x) for x in parts["confirm_hard"]],
        "train_rollout": [],
        "split": "confirm_hard",
        "task_config": "demo_randomized",
    }
    easy_path = seeds_dir / "confirm_easy.json"
    hard_path = seeds_dir / "confirm_hard.json"
    easy_path.write_text(json.dumps(easy, indent=2) + "\n", encoding="utf-8")
    hard_path.write_text(json.dumps(hard, indent=2) + "\n", encoding="utf-8")
    return easy_path, hard_path


def eval_complete(output_dir: Path, variant: str) -> bool:
    path = output_dir / "place_container_plate" / f"{variant}.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return bool((payload.get("progress") or {}).get("complete"))


def launch_eval(
    *,
    gpu: int,
    variant: str,
    ckpt: Path,
    seeds_file: Path,
    task_config: str,
    output_dir: Path,
    log_path: Path,
    extra_splits_file: Path | None = None,
    policy_seed_offset: int | None = None,
    extra_split_repeats: int = 1,
) -> subprocess.Popen:
    leaf = [
        PY,
        str(PHASE1 / "eval_per_seed.py"),
        "--task",
        "place_container_plate",
        "--task-config",
        task_config,
        "--variant",
        variant,
        "--ckpt-path",
        str(ckpt),
        "--output-dir",
        str(output_dir),
        "--seeds-file",
        str(seeds_file),
        "--id-repeats",
        "1",
        "--train-seed-count",
        "0",
        "--no-include-hard",
        "--resume",
    ]
    if extra_splits_file is not None:
        leaf.extend(
            [
                "--extra-splits-file",
                str(extra_splits_file),
                "--extra-split-repeats",
                str(extra_split_repeats),
            ]
        )
    if policy_seed_offset is not None:
        leaf.extend(["--policy-seed-offset", str(policy_seed_offset)])
    cmd = [PY, str(BRACE / "run_eval_group.py"), "--workers", str(WORKERS), "--", *leaf]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = f"{REPO}:{DP}"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    handle.write(f"\n=== launch {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} gpu={gpu} {variant} ===\n")
    handle.flush()
    return subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=handle, stderr=subprocess.STDOUT)


def make_jobs(run_dir: Path) -> list[dict[str, Any]]:
    easy_seeds, hard_seeds = build_adaptation_seeds(run_dir)
    output_dir = run_dir / "eval"
    cohort = cohort_path()
    extra_splits = None
    if cohort is not None:
        cohort_payload = json.loads(cohort.read_text(encoding="utf-8"))
        extra_splits = {
            key: [int(seed) for seed in values]
            for key, values in cohort_payload.get("cohorts", {}).items()
            if key in {"untouched_preservation", "anchor_train", "anchor_probe", "boundary"}
        }
        extra_path = run_dir / "eval_seeds" / "preservation_extra_splits.json"
        extra_path.write_text(json.dumps(extra_splits, indent=2) + "\n", encoding="utf-8")
    jobs: list[dict[str, Any]] = []

    base_ckpt = DP / "checkpoints/place_container_plate-demo_clean-200-0/600.ckpt"
    if cohort is not None and base_ckpt.is_file():
        jobs.append(
            {
                "id": "eval:base:preservation",
                "kind": "preservation",
                "method": "base",
                "seed": 0,
                "priority": 0,
                "ckpt": str(base_ckpt),
                "variants": [
                    {
                        "split": "preservation",
                        "variant": "line_a_base_preservation",
                        "seeds_file": str(BRACE / "seeds/place_container_plate_base200_line_a_seeds.json"),
                        "task_config": "demo_clean",
                        "extra_splits_file": str(run_dir / "eval_seeds/preservation_extra_splits.json"),
                        "policy_seed_offset": 4000,
                        "extra_split_repeats": 3,
                    }
                ],
            }
        )

    wave = [
        ("U0", 1),
        ("N1", 1),
        ("B1", 1),
        ("B2", 1),
        ("B3", 1),
        ("U0", 2),
        ("N1", 2),
        ("B1", 2),
        ("B2", 2),
        ("B3", 2),
        ("U0", 3),
        ("N1", 3),
        ("B1", 3),
        ("B2", 3),
        ("B3", 3),
        ("U0", 4),
        ("N1", 4),
        ("B1", 4),
        ("B2", 4),
        ("B3", 4),
        ("U0", 5),
        ("N1", 5),
        ("B1", 5),
        ("B2", 5),
        ("B3", 5),
    ]
    priority = 1
    for method, seed in wave:
        ready, info = checkpoint_ready(method, seed)
        if not ready:
            jobs.append(
                {
                    "id": f"eval:{method}:seed{seed}",
                    "kind": "adaptation",
                    "method": method,
                    "seed": seed,
                    "priority": priority,
                    "status": "blocked",
                    "blocked_reason": info.get("reason", "not_ready"),
                }
            )
            priority += 1
            continue
        variants = [
            {
                "split": "confirm_easy",
                "variant": f"line_a_{method}_seed{seed}_confirm_easy",
                "seeds_file": str(easy_seeds),
                "task_config": "demo_clean",
            },
            {
                "split": "confirm_hard",
                "variant": f"line_a_{method}_seed{seed}_confirm_hard",
                "seeds_file": str(hard_seeds),
                "task_config": "demo_randomized",
            },
        ]
        if cohort is not None and extra_splits is not None:
            variants.append(
                {
                    "split": "preservation",
                    "variant": f"line_a_{method}_seed{seed}_preservation",
                    "seeds_file": str(BRACE / "seeds/place_container_plate_base200_line_a_seeds.json"),
                    "task_config": "demo_clean",
                    "extra_splits_file": str(run_dir / "eval_seeds/preservation_extra_splits.json"),
                    "policy_seed_offset": 4000,
                    "extra_split_repeats": 3,
                }
            )
        jobs.append(
            {
                "id": f"eval:{method}:seed{seed}",
                "kind": "adaptation",
                "method": method,
                "seed": seed,
                "priority": priority,
                "status": "pending",
                "ckpt": str(checkpoint_dir(method, seed) / f"{EPOCHS}.ckpt"),
                "checkpoint_sha256": info.get("checkpoint_sha256"),
                "variants": variants,
            }
        )
        priority += 1
    return jobs


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        return False
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    if not proc.exists():
        return False
    try:
        raw = (proc / "stat").read_text()
        state = raw.split(")")[-1].split()[0]
        if state == "Z":
            return False
    except OSError:
        return False
    return True


def reclaim_orphaned_running(state: dict[str, Any]) -> int:
    """Jobs left status=running in a previous process are invisible to a fresh in-memory map."""
    n = 0
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for job in state.get("jobs") or []:
        if job.get("status") != "running":
            continue
        job["status"] = "pending"
        job.pop("pid", None)
        job.pop("gpu", None)
        job["recovery_note"] = {
            "reclaimed_from": "running",
            "at_utc": now,
            "reason": "scheduler_restart_orphaned_running",
        }
        n += 1
    return n


def main() -> int:
    os.chdir(REPO)
    run_ptr = BRACE / "runs/LATEST_place_base200_v2_line_a_pilot"
    run_dir = REPO / run_ptr.read_text(encoding="utf-8").strip()
    state_path = run_dir / "eval_state.json"
    logs = run_dir / "logs"
    output_dir = run_dir / "eval"
    gpu_ids = [int(x) for x in os.environ.get("BRACE_EVAL_GPU_IDS", " ".join(map(str, DEFAULT_EVAL_GPUS))).split()]

    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        reclaimed = reclaim_orphaned_running(state)
        if reclaimed:
            print(f"reclaimed {reclaimed} orphaned running jobs to pending", flush=True)
        state["gpus"] = gpu_ids
        jobs = make_jobs(run_dir)
        state = {
            "schema_version": 1,
            "run_dir": str(run_dir),
            "output_dir": str(output_dir),
            "gpus": gpu_ids,
            "workers_per_gpu": WORKERS,
            "cohort_path": str(cohort_path()) if cohort_path() else None,
            "jobs": jobs,
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def save() -> None:
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    save()
    running: dict[int, dict[str, Any]] = {}
    print(f"eval scheduler: {len(state['jobs'])} jobs on gpus={gpu_ids}", flush=True)

    while True:
        for gpu, slot in list(running.items()):
            if pid_alive(slot.get("pid")):
                continue
            job = slot["job"]
            variant_idx = slot["variant_idx"]
            variant = job["variants"][variant_idx]
            ok = eval_complete(output_dir, variant["variant"])
            if not ok:
                job["status"] = "failed"
                job["error"] = f"eval incomplete: {variant['variant']}"
                print(f"[failed] {job['id']} {variant['variant']}", flush=True)
                running.pop(gpu, None)
                continue
            if variant_idx + 1 < len(job["variants"]):
                next_idx = variant_idx + 1
                next_variant = job["variants"][next_idx]
                if eval_complete(output_dir, next_variant["variant"]):
                    slot["variant_idx"] = next_idx
                    continue
                log_path = logs / f"{job['id'].replace(':', '_')}_{next_variant['split']}.log"
                proc = launch_eval(
                    gpu=gpu,
                    variant=next_variant["variant"],
                    ckpt=Path(job["ckpt"]),
                    seeds_file=Path(next_variant["seeds_file"]),
                    task_config=next_variant["task_config"],
                    output_dir=output_dir,
                    log_path=log_path,
                    extra_splits_file=Path(next_variant["extra_splits_file"])
                    if next_variant.get("extra_splits_file")
                    else None,
                    policy_seed_offset=next_variant.get("policy_seed_offset"),
                    extra_split_repeats=int(next_variant.get("extra_split_repeats", 1)),
                )
                running[gpu] = {"job": job, "variant_idx": next_idx, "pid": proc.pid}
                print(f"[running] {job['id']} {next_variant['variant']} gpu={gpu}", flush=True)
                continue
            job["status"] = "completed"
            job["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            print(f"[completed] {job['id']}", flush=True)
            running.pop(gpu, None)

        # Unblock jobs whose checkpoints became static after make_jobs.
        for job in state["jobs"]:
            if job.get("status") != "blocked":
                continue
            ready, info = checkpoint_ready(job["method"], int(job["seed"]))
            if not ready:
                job["blocked_reason"] = info.get("reason", "not_ready")
                continue
            job["status"] = "pending"
            job["ckpt"] = str(checkpoint_dir(job["method"], int(job["seed"])) / f"{EPOCHS}.ckpt")
            job.pop("blocked_reason", None)
            # Ensure variants exist for jobs blocked before make_jobs filled them.
            if "variants" not in job:
                rebuilt = {
                    j["id"]: j
                    for j in make_jobs(run_dir)
                    if j.get("status") == "pending" and "variants" in j
                }
                if job["id"] in rebuilt:
                    job["variants"] = rebuilt[job["id"]]["variants"]
                    job["checkpoint_sha256"] = rebuilt[job["id"]].get("checkpoint_sha256")
            print(f"[unblocked] {job['id']} reason_cleared={info.get('reason')}", flush=True)

        pending = [j for j in state["jobs"] if j.get("status") == "pending" and "variants" in j]
        blocked = [j for j in state["jobs"] if j.get("status") == "blocked"]
        completed = [j for j in state["jobs"] if j.get("status") == "completed"]
        failed = [j for j in state["jobs"] if j.get("status") == "failed"]
        save()
        if not pending and not running:
            if blocked:
                print(
                    f"waiting on blocked={len(blocked)} completed={len(completed)} failed={len(failed)}",
                    flush=True,
                )
                time.sleep(30)
                continue
            print(
                f"eval done completed={len(completed)} failed={len(failed)} blocked={len(blocked)}",
                flush=True,
            )
            return 0 if not failed else 1
        free = [g for g in gpu_ids if g not in running]
        for gpu in free:
            pending = [j for j in state["jobs"] if j.get("status") == "pending" and "variants" in j]
            if not pending:
                break
            job = pending[0]
            # Skip variants already complete; start at first incomplete.
            start_idx = 0
            while start_idx < len(job["variants"]) and eval_complete(
                output_dir, job["variants"][start_idx]["variant"]
            ):
                start_idx += 1
            if start_idx >= len(job["variants"]):
                job["status"] = "completed"
                job["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                print(f"[completed] {job['id']} (already complete)", flush=True)
                continue
            variant = job["variants"][start_idx]
            log_path = logs / f"{job['id'].replace(':', '_')}_{variant['split']}.log"
            proc = launch_eval(
                gpu=gpu,
                variant=variant["variant"],
                ckpt=Path(job["ckpt"]),
                seeds_file=Path(variant["seeds_file"]),
                task_config=variant["task_config"],
                output_dir=output_dir,
                log_path=log_path,
                extra_splits_file=Path(variant["extra_splits_file"]) if variant.get("extra_splits_file") else None,
                policy_seed_offset=variant.get("policy_seed_offset"),
                extra_split_repeats=int(variant.get("extra_split_repeats", 1)),
            )
            job["status"] = "running"
            job["gpu"] = gpu
            job["started_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            running[gpu] = {"job": job, "variant_idx": start_idx, "pid": proc.pid}
            print(f"[running] {job['id']} {variant['variant']} gpu={gpu}", flush=True)
        save()
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
