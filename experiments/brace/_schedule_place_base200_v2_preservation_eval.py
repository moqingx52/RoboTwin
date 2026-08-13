#!/usr/bin/env python3
"""Independent Base200 Line A preservation eval scheduler.

Requires a frozen, validated cohort. Creates a *new* state file
(`preservation_eval_state.json`) and does not modify adaptation `eval_state.json`.

Jobs (26):
  - Base × 1 preservation
  - U0/N1/B1/B2/B3 × seeds 1–5 preservation
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
DEFAULT_EVAL_GPUS = (0, 1, 2, 3, 4, 5, 6, 7)
STATE_NAME = "preservation_eval_state.json"


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_cohort() -> tuple[Path, dict[str, Any]]:
    pointer = BRACE / "runs/LATEST_place_base200_line_a_preservation_cohort"
    if not pointer.is_file():
        raise SystemExit(f"missing cohort pointer: {pointer}")
    path = Path(pointer.read_text(encoding="utf-8").strip())
    if not path.is_absolute():
        path = REPO / path
    if not path.is_file():
        raise SystemExit(f"cohort missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("frozen"):
        raise SystemExit(f"cohort not frozen: {path}")
    if not payload.get("meets_min_untouched"):
        raise SystemExit(f"cohort fails meets_min_untouched: {path}")
    return path, payload


def validate_cohort_cli(cohort_path: Path) -> None:
    cmd = [
        PY,
        str(BRACE / "_validate_place_base200_line_a_cohort.py"),
        "--cohort",
        str(cohort_path),
    ]
    proc = subprocess.run(cmd, cwd=REPO)
    if proc.returncode != 0:
        raise SystemExit(f"cohort validation failed (exit={proc.returncode})")


def checkpoint_dir(method: str, seed: int) -> Path:
    name = f"place_container_plate-brace-{CHECKPOINT_LABEL}-{method}"
    return DP / "checkpoints" / f"{name}-{seed}"


def checkpoint_ready(method: str, seed: int) -> tuple[bool, dict[str, Any]]:
    cdir = checkpoint_dir(method, seed)
    ckpt = cdir / f"{EPOCHS}.ckpt"
    complete = cdir / f"{EPOCHS}.ckpt.complete"
    info: dict[str, Any] = {"checkpoint_dir": str(cdir)}
    if not ckpt.is_file() or ckpt.stat().st_size <= 0:
        return False, {**info, "reason": "missing_ckpt"}
    if not complete.is_file():
        return False, {**info, "reason": "missing_complete_marker"}
    info["checkpoint_sha256"] = file_sha256(ckpt)
    if method in ("B2", "B3"):
        feas = cdir / f"{EPOCHS}.feasibility.json"
        if not feas.is_file() or feas.stat().st_size <= 0:
            return False, {**info, "reason": "missing_feasibility"}
        info["feasibility_sha256"] = file_sha256(feas)
    return True, info


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
    extra_splits_file: Path,
    policy_seed_offset: int,
    extra_split_repeats: int,
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
        "--extra-splits-file",
        str(extra_splits_file),
        "--extra-split-repeats",
        str(extra_split_repeats),
        "--policy-seed-offset",
        str(policy_seed_offset),
    ]
    cmd = [PY, str(BRACE / "run_eval_group.py"), "--workers", str(WORKERS), "--", *leaf]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = f"{REPO}:{DP}"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    handle.write(f"\n=== launch {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} gpu={gpu} {variant} ===\n")
    handle.flush()
    return subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=handle, stderr=subprocess.STDOUT)


def make_jobs(run_dir: Path, cohort_path: Path, cohort: dict[str, Any]) -> list[dict[str, Any]]:
    seeds_dir = run_dir / "preservation_eval_seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    extra_splits = {
        key: [int(seed) for seed in values]
        for key, values in (cohort.get("cohorts") or {}).items()
        if key in {"untouched_preservation", "anchor_train", "anchor_probe", "boundary"}
    }
    extra_path = seeds_dir / "preservation_extra_splits.json"
    extra_path.write_text(json.dumps(extra_splits, indent=2) + "\n", encoding="utf-8")
    seeds_file = BRACE / "seeds/place_container_plate_base200_line_a_seeds.json"
    jobs: list[dict[str, Any]] = []

    base_ckpt = DP / "checkpoints/place_container_plate-demo_clean-200-0/600.ckpt"
    if not base_ckpt.is_file():
        raise SystemExit(f"missing Base200 checkpoint: {base_ckpt}")
    jobs.append(
        {
            "id": "pres:base",
            "kind": "preservation",
            "method": "base",
            "seed": 0,
            "priority": 0,
            "status": "pending",
            "ckpt": str(base_ckpt),
            "checkpoint_sha256": file_sha256(base_ckpt),
            "variants": [
                {
                    "split": "preservation",
                    "variant": "line_a_base_preservation",
                    "seeds_file": str(seeds_file),
                    "task_config": "demo_clean",
                    "extra_splits_file": str(extra_path),
                    "policy_seed_offset": 4000,
                    "extra_split_repeats": 3,
                }
            ],
        }
    )

    priority = 1
    for method in METHODS:
        for seed in SEEDS:
            ready, info = checkpoint_ready(method, seed)
            job: dict[str, Any] = {
                "id": f"pres:{method}:seed{seed}",
                "kind": "preservation",
                "method": method,
                "seed": seed,
                "priority": priority,
            }
            if not ready:
                job["status"] = "blocked"
                job["blocked_reason"] = info.get("reason", "not_ready")
            else:
                job["status"] = "pending"
                job["ckpt"] = str(checkpoint_dir(method, seed) / f"{EPOCHS}.ckpt")
                job["checkpoint_sha256"] = info.get("checkpoint_sha256")
                job["feasibility_sha256"] = info.get("feasibility_sha256")
                job["variants"] = [
                    {
                        "split": "preservation",
                        "variant": f"line_a_{method}_seed{seed}_preservation",
                        "seeds_file": str(seeds_file),
                        "task_config": "demo_clean",
                        "extra_splits_file": str(extra_path),
                        "policy_seed_offset": 4000,
                        "extra_split_repeats": 3,
                    }
                ]
            jobs.append(job)
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
    state_path = run_dir / STATE_NAME
    logs = run_dir / "logs_preservation"
    output_dir = run_dir / "eval_preservation"
    gpu_ids = [int(x) for x in os.environ.get("BRACE_EVAL_GPU_IDS", " ".join(map(str, DEFAULT_EVAL_GPUS))).split()]

    cohort_path, cohort = resolve_cohort()
    if not state_path.is_file():
        validate_cohort_cli(cohort_path)
        jobs = make_jobs(run_dir, cohort_path, cohort)
        state = {
            "schema_version": 1,
            "kind": "place_base200_v2_preservation_eval",
            "run_dir": str(run_dir),
            "output_dir": str(output_dir),
            "adaptation_eval_state": str(run_dir / "eval_state.json"),
            "gpus": gpu_ids,
            "workers_per_gpu": WORKERS,
            "cohort_path": str(cohort_path),
            "cohort_sha256": file_sha256(cohort_path),
            "jobs": jobs,
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": (
                "Independent of adaptation eval_state.json. Launch only after census "
                "cohort freeze + validation. Do not merge into adaptation jobs."
            ),
        }
    else:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        reclaimed = reclaim_orphaned_running(state)
        if reclaimed:
            print(f"reclaimed {reclaimed} orphaned running jobs to pending", flush=True)
        if state.get("cohort_path") != str(cohort_path):
            raise SystemExit(
                f"existing {STATE_NAME} cohort_path={state.get('cohort_path')} "
                f"differs from current pointer {cohort_path}; refuse to continue"
            )

    def save() -> None:
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    save()
    running: dict[int, dict[str, Any]] = {}
    print(
        f"preservation scheduler: {len(state['jobs'])} jobs on gpus={gpu_ids} cohort={cohort_path}",
        flush=True,
    )

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
                    extra_splits_file=Path(next_variant["extra_splits_file"]),
                    policy_seed_offset=int(next_variant["policy_seed_offset"]),
                    extra_split_repeats=int(next_variant.get("extra_split_repeats", 1)),
                )
                running[gpu] = {"job": job, "variant_idx": next_idx, "pid": proc.pid}
                print(f"[running] {job['id']} {next_variant['variant']} gpu={gpu}", flush=True)
                continue
            job["status"] = "completed"
            job["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            print(f"[completed] {job['id']}", flush=True)
            running.pop(gpu, None)

        for job in state["jobs"]:
            if job.get("status") != "blocked":
                continue
            if job["method"] == "base":
                continue
            ready, info = checkpoint_ready(job["method"], int(job["seed"]))
            if not ready:
                job["blocked_reason"] = info.get("reason", "not_ready")
                continue
            job["status"] = "pending"
            job["ckpt"] = str(checkpoint_dir(job["method"], int(job["seed"])) / f"{EPOCHS}.ckpt")
            job["checkpoint_sha256"] = info.get("checkpoint_sha256")
            job["feasibility_sha256"] = info.get("feasibility_sha256")
            job.pop("blocked_reason", None)
            if "variants" not in job:
                rebuilt = {j["id"]: j for j in make_jobs(run_dir, cohort_path, cohort) if "variants" in j}
                if job["id"] in rebuilt:
                    job["variants"] = rebuilt[job["id"]]["variants"]
            print(f"[unblocked] {job['id']}", flush=True)

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
                f"preservation eval done completed={len(completed)} failed={len(failed)}",
                flush=True,
            )
            return 0 if not failed else 1
        free = [g for g in gpu_ids if g not in running]
        for gpu in free:
            pending = [j for j in state["jobs"] if j.get("status") == "pending" and "variants" in j]
            if not pending:
                break
            job = pending[0]
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
                extra_splits_file=Path(variant["extra_splits_file"]),
                policy_seed_offset=int(variant["policy_seed_offset"]),
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
