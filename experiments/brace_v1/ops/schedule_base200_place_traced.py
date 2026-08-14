#!/usr/bin/env python3
"""Resumable Base200 place traced collection with a fixed 15-shard layout.

Waits for place_container_plate Base200 frozen eval (easy+hard) to complete,
then rolls freed GPUs into shard groups of 3 workers each without changing
num_shards mid-run. Never touches GPUs outside the configured pool.

Hard locks:
- place gate = merged easy+hard complete + Base200 provenance valid; never SR-gated
- rollout_train seeds cite multitask_amendment.v2.1 with path/SHA and partition
  disjointness vs easy/hard/census/anchor/train
- free-GPU detection uses nvidia-smi compute apps (+ CVD as corroboration), not
  only scheduler state
- merge/audit only after completeness over shards 0..14
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AMENDMENT = Path("experiments/brace/multitask_amendment.v2.1.json")
DEFAULT_PROTOCOL = Path("experiments/brace/multitask_protocol.v2.json")
IDLE_GPU_MEMORY_MIB = 400
CONDA_PYTHON = Path("/root/miniconda/envs/RoboTwin/bin/python")


def python_bin() -> str:
    if CONDA_PYTHON.is_file():
        return str(CONDA_PYTHON)
    return sys.executable


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def place_eval_gate(
    eval_run_dir: Path,
    *,
    protocol_path: Path,
    ckpt_path: Path,
) -> tuple[bool, dict[str, Any]]:
    """Completion+provenance gate only. Success rates are recorded, never deciding."""
    details: dict[str, Any] = {
        "rule": "easy_and_hard_merged_complete_and_base200_provenance_valid",
        "uses_success_rate_threshold": False,
        "traced_launch_conditioned_on_eval_sr": False,
        "variants": {},
        "provenance_ok": False,
    }
    protocol_sha = file_sha256(protocol_path)
    prov_path = ckpt_path.with_name("brace_base200_provenance.json")
    if not ckpt_path.is_file() or not prov_path.is_file():
        details["error"] = "missing Base200 ckpt or provenance"
        return False, details
    prov = read_json(prov_path)
    details["provenance"] = {
        "path": str(prov_path),
        "protocol_sha256": prov.get("protocol_sha256"),
        "expected_protocol_sha256": protocol_sha,
    }
    if prov.get("protocol_sha256") != protocol_sha:
        details["error"] = "Base200 provenance protocol hash mismatch"
        return False, details
    details["provenance_ok"] = True

    all_complete = True
    for variant in ("base200_confirm_easy", "base200_confirm_hard"):
        path = eval_run_dir / "results" / "place_container_plate" / f"{variant}.json"
        row: dict[str, Any] = {"path": str(path), "present": path.is_file(), "complete": False}
        if path.is_file():
            payload = read_json(path)
            progress = payload.get("progress") or {}
            row["complete"] = bool(progress.get("complete"))
            # Record SR for observability only — never gate on it.
            heldout = ((payload.get("splits") or {}).get("id_heldout") or {})
            row["mean_sr_observational"] = heldout.get("mean_sr")
            row["episodes_observational"] = heldout.get("episodes")
        details["variants"][variant] = row
        all_complete = all_complete and bool(row["complete"])
    return all_complete and details["provenance_ok"], details


def materialize_seeds(
    seed_manifest: Path,
    out_path: Path,
    train_seed_file: Path,
    amendment_path: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    if not amendment_path.is_file():
        raise SystemExit(f"missing amendment: {amendment_path}")
    sidecar = Path(str(amendment_path) + ".sha256")
    amendment_sha = file_sha256(amendment_path)
    if sidecar.is_file():
        side_text = sidecar.read_text(encoding="utf-8").strip().split()
        if not side_text or side_text[0] != amendment_sha:
            raise SystemExit(f"amendment SHA256 sidecar mismatch: {sidecar}")
    amendment = read_json(amendment_path)
    if amendment.get("amendment_revision") != "brace.multitask.v2.1":
        raise SystemExit("expected brace.multitask.v2.1 amendment")
    applies = amendment.get("applies_to") or {}
    protocol_sha = file_sha256(protocol_path)
    if applies.get("protocol_sha256") != protocol_sha:
        raise SystemExit(
            f"amendment protocol_sha256 mismatch: {applies.get('protocol_sha256')} != {protocol_sha}"
        )
    cited = amendment.get("line_a_base200_traced_seeds") or {}
    if Path(cited.get("source_manifest", "")) != seed_manifest:
        raise SystemExit("amendment source_manifest does not match --seed-manifest")

    manifest = read_json(seed_manifest)
    manifest_sha = file_sha256(seed_manifest)
    if cited.get("source_manifest_sha256") != manifest_sha:
        raise SystemExit(
            f"source manifest SHA mismatch: amendment={cited.get('source_manifest_sha256')} "
            f"actual={manifest_sha}"
        )
    parts = manifest["partitions"]
    seeds = [int(x) for x in parts["rollout_train"]]
    if len(seeds) != int(cited.get("expected_count", 100)):
        raise SystemExit(f"expected {cited.get('expected_count')} rollout_train seeds, got {len(seeds)}")

    train = {int(x) for x in train_seed_file.read_text(encoding="utf-8").split()}
    checks = {
        "confirm_easy": set(map(int, parts["confirm_easy"])),
        "confirm_hard": set(map(int, parts["confirm_hard"])),
        "census_candidate": set(map(int, parts["census_candidate"])),
        "anchor_candidate": set(map(int, parts["anchor_candidate"])),
        "base200_train_seed_txt": train,
    }
    seed_set = set(seeds)
    overlaps = {name: sorted(seed_set & other)[:5] for name, other in checks.items() if seed_set & other}
    if overlaps:
        raise SystemExit(f"rollout_train overlaps other partitions/train: {overlaps}")

    meta = {
        "schema_version": 1,
        "task": "place_container_plate",
        "split": "rollout_train",
        "train_rollout": seeds,
        "source_manifest": str(seed_manifest),
        "source_manifest_sha256": manifest_sha,
        "source_manifest_status": manifest.get("status"),
        "source_partition": "partitions.rollout_train",
        "amendment": str(amendment_path),
        "amendment_revision": amendment.get("amendment_revision"),
        "amendment_sha256": amendment_sha,
        "protocol": str(protocol_path),
        "protocol_revision": "brace.multitask.v2",
        "protocol_sha256": protocol_sha,
        "disjoint_from": {
            name: {"count": len(other), "overlap": 0} for name, other in checks.items()
        },
        "notes": (
            "Base200 v2 Line A traced corpus. Seeds are the multitask_v1 "
            "rollout_train partition cited by brace.multitask.v2.1; not a silent v1 dependency."
        ),
    }
    write_json(out_path, meta)
    return meta


def proc_cuda_visible(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    for item in raw.split(b"\0"):
        if item.startswith(b"CUDA_VISIBLE_DEVICES="):
            return item.decode("utf-8", errors="replace").split("=", 1)[1]
    return None


def proc_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ""


def nvidia_gpu_index_map() -> dict[str, int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        text=True,
    )
    mapping: dict[str, int] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            mapping[parts[1]] = int(parts[0])
    return mapping


def nvidia_memory_used_mib() -> dict[int, int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    mem: dict[int, int] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            mem[int(parts[0])] = int(float(parts[1]))
    return mem


def nvidia_compute_pids_by_gpu() -> dict[int, set[int]]:
    uuid_to_index = nvidia_gpu_index_map()
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )
    by_gpu: dict[int, set[int]] = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        uuid, pid_s = parts[0], parts[1]
        if uuid not in uuid_to_index:
            continue
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        by_gpu.setdefault(uuid_to_index[uuid], set()).add(pid)
    return by_gpu


def cvd_busy_gpus(pool: list[int], reserved_pids: set[int]) -> set[int]:
    """Corroborating signal from CUDA_VISIBLE_DEVICES on known simulator processes."""
    busy: set[int] = set()
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", "eval_per_seed.py|run_eval_group.py|collect_traced_rollouts.py|train.py"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return busy
    for line in out.splitlines():
        try:
            pid = int(line.split(None, 1)[0])
        except ValueError:
            continue
        if pid in reserved_pids or pid == os.getpid():
            continue
        cvd = proc_cuda_visible(pid)
        if cvd is None:
            continue
        token = cvd.strip()
        if not token or "," in token:
            continue
        try:
            gpu = int(token)
        except ValueError:
            continue
        if gpu in pool:
            busy.add(gpu)
    return busy


def free_gpus_in_pool(pool: list[int], reserved_pids: set[int]) -> tuple[list[int], dict[str, Any]]:
    """Free iff nvidia-smi shows no non-reserved compute apps and memory near idle."""
    compute = nvidia_compute_pids_by_gpu()
    mem = nvidia_memory_used_mib()
    cvd_busy = cvd_busy_gpus(pool, reserved_pids)
    details: dict[str, Any] = {"gpus": {}, "cvd_busy": sorted(cvd_busy)}
    free: list[int] = []
    for gpu in pool:
        apps = set(compute.get(gpu, set()))
        own_apps = sorted(apps & reserved_pids)
        foreign_apps = sorted(apps - reserved_pids)
        used = int(mem.get(gpu, 0))
        # Authoritative: any foreign compute app => busy (covers orphan click_easy).
        # Own workers also mark the GPU busy — reserved_pids must NOT make a GPU
        # look free, or each poll restacks another 3-worker group onto it.
        busy_reason = None
        if foreign_apps:
            busy_reason = "nvidia_compute_apps"
        elif own_apps:
            busy_reason = "own_scheduler_workers"
        elif gpu in cvd_busy:
            busy_reason = "cuda_visible_devices"
        elif used > IDLE_GPU_MEMORY_MIB and not apps:
            # Residual context without listed apps — do not steal.
            busy_reason = "memory_above_idle_without_apps"
        details["gpus"][str(gpu)] = {
            "memory_used_mib": used,
            "compute_pids": sorted(apps),
            "own_pids": own_apps,
            "foreign_pids": foreign_apps,
            "foreign_cmdlines": [proc_cmdline(pid)[:160] for pid in foreign_apps[:4]],
            "busy_reason": busy_reason,
        }
        if busy_reason is None:
            free.append(gpu)
    return free, details


def shard_done(task_dir: Path, shard_id: int, num_shards: int) -> bool:
    stats = task_dir / f"seed_stats_shard_{shard_id:02d}_of_{num_shards:02d}.json"
    manifest = task_dir / f"manifest_shard_{shard_id:02d}_of_{num_shards:02d}.jsonl"
    return stats.is_file() and manifest.is_file()


def assert_shard_layout_complete(
    task_dir: Path,
    *,
    num_shards: int,
    seeds: list[int],
    rollouts_per_seed: int,
) -> dict[str, Any]:
    """Pre-merge completeness: exact shards 0..N-1, unified num_shards, unique keys, HDF5 paths."""
    expected_manifests = [
        task_dir / f"manifest_shard_{i:02d}_of_{num_shards:02d}.jsonl" for i in range(num_shards)
    ]
    missing_files = [str(p) for p in expected_manifests if not p.is_file()]
    unexpected = sorted(
        p.name
        for p in task_dir.glob("manifest_shard_*_of_*.jsonl")
        if p not in expected_manifests
    )
    rows: list[dict[str, Any]] = []
    for path in expected_manifests:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    keys = [(int(r["env_seed"]), int(r["rollout_id"])) for r in rows]
    counts = Counter(keys)
    expected = {(s, r) for s in seeds for r in range(rollouts_per_seed)}
    missing_keys = sorted(expected - set(keys))
    extra_keys = sorted(set(keys) - expected)
    duplicates = sorted(k for k, c in counts.items() if c > 1)
    missing_success_hdf5 = 0
    missing_failure_hdf5 = 0
    for row in rows:
        if row.get("success"):
            path_value = row.get("hdf5_path")
            path = Path(path_value) if path_value else None
            if path is not None and not path.is_absolute():
                path = REPO_ROOT / path
            if path is None or not path.is_file():
                missing_success_hdf5 += 1
        else:
            path_value = row.get("failure_hdf5_path")
            path = Path(path_value) if path_value else None
            if path is not None and not path.is_absolute():
                path = REPO_ROOT / path
            if path is None or not path.is_file():
                missing_failure_hdf5 += 1
    report = {
        "num_shards": num_shards,
        "expected_shard_ids": list(range(num_shards)),
        "missing_manifest_files": missing_files,
        "unexpected_manifest_files": unexpected,
        "rows": len(rows),
        "expected_rows": len(expected),
        "missing_keys": len(missing_keys),
        "extra_keys": len(extra_keys),
        "duplicate_keys": len(duplicates),
        "missing_success_hdf5": missing_success_hdf5,
        "missing_failure_hdf5": missing_failure_hdf5,
    }
    ok = (
        not missing_files
        and not unexpected
        and not missing_keys
        and not extra_keys
        and not duplicates
        and missing_success_hdf5 == 0
        and missing_failure_hdf5 == 0
    )
    report["ok"] = ok
    if not ok:
        raise SystemExit(f"traced completeness failed: {json.dumps(report)}")
    return report


def launch_shard(
    *,
    gpu: int,
    shard_id: int,
    num_shards: int,
    seeds_file: Path,
    rollout_dir: Path,
    log_dir: Path,
    expert_data_num: int,
    checkpoint_num: int,
    rollouts_per_seed: int,
    action_dim: int,
) -> subprocess.Popen[bytes]:
    log_path = log_dir / f"place_container_plate_shard{shard_id:02d}_of_{num_shards:02d}.log"
    cmd = [
        python_bin(),
        "experiments/brace/collect_traced_rollouts.py",
        "--task",
        "place_container_plate",
        "--task-config",
        "demo_brace_trace",
        "--seeds-file",
        str(seeds_file),
        "--output-dir",
        str(rollout_dir),
        "--expert-data-num",
        str(expert_data_num),
        "--checkpoint-num",
        str(checkpoint_num),
        "--train-seed",
        "0",
        "--rollouts-per-seed",
        str(rollouts_per_seed),
        "--action-dim",
        str(action_dim),
        "--shard-id",
        str(shard_id),
        "--num-shards",
        str(num_shards),
        "--snapshots-per-trajectory",
        "3",
        "--save-failures",
        "--resume",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT}/policy/DP" + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    proc._brace_log_f = log_f  # type: ignore[attr-defined]
    return proc


def close_log(proc: subprocess.Popen[bytes]) -> None:
    handle = getattr(proc, "_brace_log_f", None)
    if handle is not None:
        try:
            handle.close()
        except Exception:
            pass


def verify_and_merge(
    *,
    rollout_dir: Path,
    seeds_file: Path,
    num_shards: int,
    rollouts_per_seed: int,
) -> None:
    py = python_bin()
    subprocess.check_call(
        [
            py,
            "experiments/brace/verify_traced_rollouts.py",
            "--tasks",
            "place_container_plate",
            "--rollout-dir",
            str(rollout_dir),
            "--seeds-file",
            str(seeds_file),
            "--rollouts-per-seed",
            str(rollouts_per_seed),
            "--num-shards",
            str(num_shards),
            "--require-failures",
            "--workers",
            os.environ.get("BRACE_VERIFY_WORKERS", "96"),
        ],
        cwd=str(REPO_ROOT),
    )
    subprocess.check_call(
        [
            py,
            "experiments/phase1/merge_rollout_shards.py",
            "--task",
            "place_container_plate",
            "--task-config",
            "demo_brace_trace",
            "--seeds-file",
            str(seeds_file),
            "--rollouts-per-seed",
            str(rollouts_per_seed),
            "--rollout-dir",
            str(rollout_dir),
        ],
        cwd=str(REPO_ROOT),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-run-dir", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument(
        "--rollout-dir",
        type=Path,
        default=Path("experiments/brace/rollouts_traced_base200_v2"),
    )
    parser.add_argument(
        "--seed-manifest",
        type=Path,
        default=Path("experiments/brace/seeds/multitask_v1/place_container_plate.json"),
    )
    parser.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=Path("policy/DP/checkpoints/place_container_plate-demo_clean-200-0/600.ckpt"),
    )
    parser.add_argument("--gpus", nargs="+", type=int, default=[2, 4, 5, 6, 7])
    parser.add_argument("--num-shards", type=int, default=15)
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--expert-data-num", type=int, default=200)
    parser.add_argument("--checkpoint-num", type=int, default=600)
    parser.add_argument("--rollouts-per-seed", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--log-dir", type=Path, default=Path("experiments/brace/logs/traced_base200_v2"))
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    if args.num_shards % args.workers_per_gpu != 0:
        raise SystemExit("num_shards must be divisible by workers_per_gpu for fixed groups")
    if args.num_shards != 15:
        print(f"WARNING: num_shards={args.num_shards} (protocol default is 15)", flush=True)

    os.chdir(REPO_ROOT)
    lock_path = args.state.with_suffix(".lock")
    if lock_path.exists():
        try:
            old_pid = int(lock_path.read_text(encoding="utf-8").strip())
            os.kill(old_pid, 0)
            raise SystemExit(f"another scheduler holds {lock_path} (pid={old_pid})")
        except (ProcessLookupError, ValueError, OSError):
            pass
    lock_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")

    def cleanup(_signum=None, _frame=None) -> None:
        try:
            if lock_path.is_file() and lock_path.read_text().strip() == str(os.getpid()):
                lock_path.unlink(missing_ok=True)
        finally:
            if _signum is not None:
                raise SystemExit(128 + int(_signum))

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    seeds_file = args.rollout_dir / "place_container_plate" / "base200_v2_rollout_train_seeds.json"
    seed_meta = materialize_seeds(
        args.seed_manifest,
        seeds_file,
        Path("data/place_container_plate/demo_clean/seed.txt"),
        args.amendment,
        args.protocol,
    )
    provenance_path = args.rollout_dir / "place_container_plate" / "brace_base200_traced_provenance.json"
    write_json(
        provenance_path,
        {
            "schema_version": 1,
            "stage": "base200_line_a_traced",
            "task": "place_container_plate",
            "num_shards": args.num_shards,
            "workers_per_gpu": args.workers_per_gpu,
            "gpus": args.gpus,
            "rollout_dir": str(args.rollout_dir),
            "eval_run_dir": str(args.eval_run_dir),
            "ckpt_path": str(args.ckpt_path),
            "seeds": seed_meta,
            "place_eval_gate": {
                "rule": "easy_and_hard_merged_complete_and_base200_provenance_valid",
                "uses_success_rate_threshold": False,
                "traced_launch_conditioned_on_eval_sr": False,
            },
            "priority_after_shards": [
                "place_replay_audit",
                "place_branch_verification_and_matched_random",
                "remaining_base200_frozen_eval_including_click_hard",
            ],
        },
    )
    task_dir = args.rollout_dir / "place_container_plate"
    task_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(seed_meta["train_rollout"])

    if args.state.is_file():
        state = read_json(args.state)
    else:
        state = {
            "schema_version": 2,
            "task": "place_container_plate",
            "num_shards": args.num_shards,
            "workers_per_gpu": args.workers_per_gpu,
            "gpus": args.gpus,
            "rollout_dir": str(args.rollout_dir),
            "seeds_file": str(seeds_file),
            "eval_run_dir": str(args.eval_run_dir),
            "amendment": str(args.amendment),
            "place_eval_gate_passed": False,
            "place_eval_gate_details": None,
            "shards": {
                str(i): {"status": "pending", "gpu": None, "pids": []}
                for i in range(args.num_shards)
            },
            "verify_done": False,
        }
        write_json(args.state, state)

    if int(state["num_shards"]) != args.num_shards:
        raise SystemExit(
            f"state num_shards={state['num_shards']} != CLI {args.num_shards}; "
            "do not change layout mid-run"
        )

    print(
        json.dumps(
            {
                "event": "scheduler_start",
                "state": str(args.state),
                "rollout_dir": str(args.rollout_dir),
                "num_shards": args.num_shards,
                "gpus": args.gpus,
                "amendment": str(args.amendment),
                "source_manifest_sha256": seed_meta["source_manifest_sha256"],
            }
        ),
        flush=True,
    )

    # Self-check: orphan click on GPU 6 must appear busy while it is running.
    free0, occ0 = free_gpus_in_pool(args.gpus, reserved_pids=set())
    print(json.dumps({"event": "gpu_occupancy_bootstrap", "free": free0, "details": occ0}), flush=True)

    live: dict[int, subprocess.Popen[bytes]] = {}

    try:
        while True:
            if not state["place_eval_gate_passed"]:
                ok, details = place_eval_gate(
                    args.eval_run_dir,
                    protocol_path=args.protocol,
                    ckpt_path=args.ckpt_path,
                )
                state["place_eval_gate_details"] = details
                write_json(args.state, state)
                if ok:
                    state["place_eval_gate_passed"] = True
                    write_json(args.state, state)
                    print(json.dumps({"event": "place_eval_gate_passed", "details": details}), flush=True)
                else:
                    print(json.dumps({"event": "waiting_place_eval", "details": details}), flush=True)
                    time.sleep(args.poll_seconds)
                    continue

            reserved_pids = set(live.keys())
            for meta in state["shards"].values():
                for pid in meta.get("pids") or []:
                    reserved_pids.add(int(pid))

            for shard_s, meta in state["shards"].items():
                shard_id = int(shard_s)
                if meta["status"] == "completed":
                    continue
                if shard_done(task_dir, shard_id, args.num_shards):
                    meta["status"] = "completed"
                    meta["gpu"] = None
                    meta["pids"] = []
                    continue
                if meta["status"] == "running":
                    still = []
                    for pid in meta.get("pids") or []:
                        proc = live.get(pid)
                        if proc is None:
                            try:
                                os.kill(pid, 0)
                                still.append(pid)
                            except OSError:
                                pass
                            continue
                        rc = proc.poll()
                        if rc is None:
                            still.append(pid)
                        else:
                            close_log(proc)
                            live.pop(pid, None)
                            if rc != 0 and not shard_done(task_dir, shard_id, args.num_shards):
                                meta["status"] = "failed"
                                meta["returncode"] = rc
                                print(
                                    json.dumps(
                                        {
                                            "event": "shard_failed",
                                            "shard": shard_id,
                                            "pid": pid,
                                            "rc": rc,
                                        }
                                    ),
                                    flush=True,
                                )
                    meta["pids"] = still
                    if not still and meta["status"] == "running":
                        if shard_done(task_dir, shard_id, args.num_shards):
                            meta["status"] = "completed"
                            meta["gpu"] = None
                        else:
                            meta["status"] = "failed"
                    if meta["status"] != "running":
                        meta["gpu"] = None

            write_json(args.state, state)

            pending = [int(s) for s, m in state["shards"].items() if m["status"] in ("pending", "failed")]
            running = [int(s) for s, m in state["shards"].items() if m["status"] == "running"]
            completed = [int(s) for s, m in state["shards"].items() if m["status"] == "completed"]
            free, occ = free_gpus_in_pool(args.gpus, reserved_pids)
            print(
                json.dumps(
                    {
                        "event": "status",
                        "pending": len(pending),
                        "running": len(running),
                        "completed": len(completed),
                        "free_gpus": free,
                        "busy_reasons": {
                            g: occ["gpus"][g]["busy_reason"]
                            for g in occ["gpus"]
                            if occ["gpus"][g]["busy_reason"]
                        },
                    }
                ),
                flush=True,
            )

            if len(completed) == args.num_shards and not running:
                break

            group_size = args.workers_per_gpu
            launched_any = False
            for gpu in free:
                chosen = None
                for start in range(0, args.num_shards, group_size):
                    group = list(range(start, start + group_size))
                    statuses = [state["shards"][str(s)]["status"] for s in group]
                    if any(st == "running" for st in statuses):
                        continue
                    need = [s for s, st in zip(group, statuses) if st in ("pending", "failed")]
                    if need:
                        chosen = need
                        break
                if chosen is None:
                    break
                pids = []
                for shard_id in chosen:
                    proc = launch_shard(
                        gpu=gpu,
                        shard_id=shard_id,
                        num_shards=args.num_shards,
                        seeds_file=seeds_file,
                        rollout_dir=args.rollout_dir,
                        log_dir=args.log_dir,
                        expert_data_num=args.expert_data_num,
                        checkpoint_num=args.checkpoint_num,
                        rollouts_per_seed=args.rollouts_per_seed,
                        action_dim=args.action_dim,
                    )
                    live[proc.pid] = proc
                    reserved_pids.add(proc.pid)
                    pids.append(proc.pid)
                    state["shards"][str(shard_id)] = {
                        "status": "running",
                        "gpu": gpu,
                        "pids": [proc.pid],
                        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                print(
                    json.dumps(
                        {"event": "launch_group", "gpu": gpu, "shards": chosen, "pids": pids}
                    ),
                    flush=True,
                )
                write_json(args.state, state)
                launched_any = True

            if not launched_any and not running and pending:
                print(json.dumps({"event": "waiting_free_gpu", "occupancy": occ}), flush=True)

            time.sleep(args.poll_seconds)

        completeness = assert_shard_layout_complete(
            task_dir,
            num_shards=args.num_shards,
            seeds=seeds,
            rollouts_per_seed=args.rollouts_per_seed,
        )
        state["completeness"] = completeness
        write_json(args.state, state)
        print(json.dumps({"event": "completeness_ok", "report": completeness}), flush=True)

        if not args.skip_verify and not state.get("verify_done"):
            print(json.dumps({"event": "verify_merge_start"}), flush=True)
            verify_and_merge(
                rollout_dir=args.rollout_dir,
                seeds_file=seeds_file,
                num_shards=args.num_shards,
                rollouts_per_seed=args.rollouts_per_seed,
            )
            state["verify_done"] = True
            write_json(args.state, state)
            print(json.dumps({"event": "verify_merge_done"}), flush=True)

        print(json.dumps({"event": "scheduler_done", "rollout_dir": str(args.rollout_dir)}), flush=True)
        return 0
    finally:
        cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
