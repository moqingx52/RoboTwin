#!/usr/bin/env python3
"""Reshard B3 seed4/5 preservation tails across idle GPUs, resume existing rows.

Keeps B3 seed1/3 on their current GPUs. Uses GPUs 0,2,5 for seed4 and 4,6,7 for
seed5 (3 GPUs × 3 shards each).
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/workspace/RoboTwin")
PY = "/root/miniconda/envs/RoboTwin/bin/python"
PHASE1 = REPO / "experiments" / "phase1"
RUN = REPO / "experiments/brace/runs/20260812T031200Z_place_base200_v2_line_a_pilot"
OUT = RUN / "eval_preservation"
TASK_DIR = OUT / "place_container_plate"
SEEDS = REPO / "experiments/brace/seeds/place_container_plate_base200_line_a_seeds.json"
EXTRA = RUN / "preservation_eval_seeds/preservation_extra_splits.json"
LOGS = RUN / "logs_preservation"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(PHASE1) not in sys.path:
    sys.path.insert(0, str(PHASE1))

from eval_per_seed import build_work_items, summarize, work_item_key  # noqa: E402
from common import read_json, write_json_atomic  # noqa: E402

JOBS = [
    {
        "seed": 4,
        "variant": "line_a_B3_seed4_preservation",
        "ckpt": REPO / "policy/DP/checkpoints/place_container_plate-brace-place_base200_v2-B3-4/10.ckpt",
        "gpus": [0, 2, 3],
    },
    {
        "seed": 5,
        "variant": "line_a_B3_seed5_preservation",
        "ckpt": REPO / "policy/DP/checkpoints/place_container_plate-brace-place_base200_v2-B3-5/10.ckpt",
        "gpus": [4, 5, 6, 7],
    },
]
WORKERS_PER_GPU = 3


def collect_old_rows(variant: str) -> tuple[list[dict], dict]:
    paths = sorted(TASK_DIR.glob(f"{variant}_shard_*_of_03.json"))
    if not paths:
        raise SystemExit(f"missing old shards for {variant}")
    rows = []
    seen = set()
    meta = None
    for path in paths:
        payload = read_json(path)
        if meta is None:
            meta = payload
        for row in payload.get("rows") or []:
            key = work_item_key(row["split"], row["env_seed"], row["repeat"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows, meta


def redistribute(variant: str, rows: list[dict], meta: dict, num_shards: int) -> int:
    seed_payload = read_json(SEEDS)
    seed_payload = dict(seed_payload)
    seed_payload["train_rollout"] = []
    extra = {str(k): [int(x) for x in v] for k, v in read_json(EXTRA).items()}
    items = build_work_items(
        seed_payload,
        [],
        1,
        3,
        0,
        extra,
        3,
        census_candidate_split=False,
        census_candidate_id_count=None,
    )
    item_shard = {work_item_key(*item): index % num_shards for index, item in enumerate(items)}
    distributed = [[] for _ in range(num_shards)]
    for row in rows:
        key = work_item_key(row["split"], row["env_seed"], row["repeat"])
        if key not in item_shard:
            raise SystemExit(f"unexpected row {key} in {variant}")
        distributed[item_shard[key]].append(row)
    for shard, shard_rows in enumerate(distributed):
        expected = sum(1 for index in range(len(items)) if index % num_shards == shard)
        payload = copy.deepcopy(meta)
        payload["rows"] = shard_rows
        payload["splits"] = {
            split: summarize(shard_rows, split)
            for split in sorted({row["split"] for row in shard_rows} or {"id_heldout"})
        }
        payload["progress"] = {
            "complete": len(shard_rows) == expected,
            "completed_episodes": len(shard_rows),
            "id_repeats": 1,
            "train_repeats": 3,
            "hard_repeats": 0,
            "extra_split_repeats": 3,
            "policy_seed_offset": 4000,
            "shard_id": shard,
            "num_shards": num_shards,
        }
        out = TASK_DIR / f"{variant}_shard_{shard:02d}_of_{num_shards:02d}.json"
        write_json_atomic(out, payload)
        print(f"wrote {out.name} rows={len(shard_rows)} expected={expected} complete={len(shard_rows)==expected}", flush=True)
    for old in TASK_DIR.glob(f"{variant}_shard_*_of_03.json"):
        old.unlink()
        print(f"removed {old.name}", flush=True)
    return len(rows)


def launch_job(job: dict) -> list[subprocess.Popen]:
    gpus = job["gpus"]
    num_shards = len(gpus) * WORKERS_PER_GPU
    variant = job["variant"]
    rows, meta = collect_old_rows(variant)
    n = redistribute(variant, rows, meta, num_shards)
    print(f"{variant} resumed_rows={n} shards={num_shards} gpus={gpus}", flush=True)
    leaf = [
        PY,
        str(PHASE1 / "eval_per_seed.py"),
        "--task",
        "place_container_plate",
        "--task-config",
        "demo_clean",
        "--variant",
        variant,
        "--ckpt-path",
        str(job["ckpt"]),
        "--output-dir",
        str(OUT),
        "--seeds-file",
        str(SEEDS),
        "--id-repeats",
        "1",
        "--train-seed-count",
        "0",
        "--no-include-hard",
        "--resume",
        "--extra-splits-file",
        str(EXTRA),
        "--extra-split-repeats",
        "3",
        "--policy-seed-offset",
        "4000",
        "--num-shards",
        str(num_shards),
    ]
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = f"{REPO}:{REPO}/policy/DP"
    procs = []
    LOGS.mkdir(parents=True, exist_ok=True)
    for gi, gpu in enumerate(gpus):
        for local in range(WORKERS_PER_GPU):
            shard = gi * WORKERS_PER_GPU + local
            env = env_base.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_path = LOGS / f"{variant}_reshard_{shard:02d}_of_{num_shards:02d}.log"
            handle = log_path.open("w", encoding="utf-8")
            cmd = leaf + ["--shard-id", str(shard)]
            proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=handle, stderr=subprocess.STDOUT)
            procs.append(proc)
            print(f"launch {variant} shard={shard:02d} gpu={gpu} pid={proc.pid}", flush=True)
    return procs


def merge_variant(variant: str, num_shards: int) -> None:
    cmd = [
        PY,
        str(PHASE1 / "merge_eval_shards.py"),
        "--task",
        "place_container_plate",
        "--task-config",
        "demo_clean",
        "--variant",
        variant,
        "--output-dir",
        str(OUT),
        "--num-shards",
        str(num_shards),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO), check=False)
    if proc.returncode != 0:
        raise SystemExit(f"merge failed {variant} exit={proc.returncode}")
    payload = read_json(TASK_DIR / f"{variant}.json")
    complete = bool((payload.get("progress") or {}).get("complete"))
    n_rows = len(payload.get("rows") or [])
    print(f"merged {variant} complete={complete} n_rows={n_rows}", flush=True)
    if not complete or n_rows != 328:
        raise SystemExit(f"incomplete merge {variant}")


def variant_complete(variant: str) -> bool:
    path = TASK_DIR / f"{variant}.json"
    if not path.is_file():
        return False
    payload = read_json(path)
    return bool((payload.get("progress") or {}).get("complete")) and len(payload.get("rows") or []) == 328


def wait_parked(variants: list[str]) -> None:
    while True:
        done = [v for v in variants if variant_complete(v)]
        pending = [v for v in variants if v not in done]
        print(
            f"[{time.strftime('%H:%M:%S')}] parked complete={done} waiting={pending}",
            flush=True,
        )
        if not pending:
            return
        time.sleep(20)


def mark_jobs_completed(job_ids: list[str]) -> None:
    path = RUN / "preservation_eval_state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    found = set()
    for job in state.get("jobs") or []:
        if job.get("id") not in job_ids:
            continue
        found.add(job["id"])
        job["status"] = "completed"
        job["finished_at_utc"] = now
        job["recovery_note"] = {
            "at_utc": now,
            "reason": "manual_tail_reshard_all_gpus",
            "note": "B3 seed4/5 redistributed across idle GPUs; seed1/3 kept on original GPUs",
        }
        job.pop("error", None)
    missing = [jid for jid in job_ids if jid not in found]
    if missing:
        raise SystemExit(f"jobs missing from preservation_eval_state.json: {missing}")
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"marked completed {sorted(found)}", flush=True)


def main() -> int:
    os.chdir(REPO)
    all_procs: list[tuple[str, int, subprocess.Popen]] = []
    for job in JOBS:
        procs = launch_job(job)
        num_shards = len(job["gpus"]) * WORKERS_PER_GPU
        for proc in procs:
            all_procs.append((job["variant"], num_shards, proc))
    while True:
        alive = [p for _, _, p in all_procs if p.poll() is None]
        failed = [(v, p.returncode) for v, _, p in all_procs if p.poll() not in (None, 0)]
        print(
            f"[{time.strftime('%H:%M:%S')}] reshard alive={len(alive)} failed={len(failed)}",
            flush=True,
        )
        if failed:
            print(f"failures {failed}", flush=True)
            return 1
        if not alive:
            break
        time.sleep(30)
    for job in JOBS:
        merge_variant(job["variant"], len(job["gpus"]) * WORKERS_PER_GPU)
    wait_parked(
        [
            "line_a_B3_seed1_preservation",
            "line_a_B3_seed3_preservation",
        ]
    )
    mark_jobs_completed(
        [
            "pres:B3:seed1",
            "pres:B3:seed3",
            "pres:B3:seed4",
            "pres:B3:seed5",
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
