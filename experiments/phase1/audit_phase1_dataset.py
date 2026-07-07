#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import VARIANTS, iter_jsonl, read_json, repo_path, write_json


DEFAULT_VARIANTS = ("expert_only", "success", "seed_balanced", "difficulty_weighted")


def rel_or_abs(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return repo_path(path)


def pct(values, q):
    if values.size == 0:
        return math.nan
    return float(np.percentile(values, q))


def stats_1d(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "p01": math.nan,
            "p50": math.nan,
            "p99": math.nan,
            "max": math.nan,
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p01": pct(values, 1),
        "p50": pct(values, 50),
        "p99": pct(values, 99),
        "max": float(values.max()),
    }


def array_summary(arr, max_rows=None):
    data = arr[:]
    if max_rows is not None and data.shape[0] > max_rows:
        idx = np.linspace(0, data.shape[0] - 1, max_rows).astype(np.int64)
        data = data[idx]
    flat = data.reshape(-1, data.shape[-1])
    per_dim = []
    for i in range(flat.shape[1]):
        dim = flat[:, i]
        dim_stats = stats_1d(dim)
        dim_stats["dim"] = i
        dim_stats["near_constant"] = bool(np.nan_to_num(dim_stats["std"]) < 1e-8)
        per_dim.append(dim_stats)
    return per_dim


def episode_slices(episode_ends):
    start = 0
    for end in episode_ends:
        end = int(end)
        yield start, end
        start = end


def summarize_sources(root):
    episode_ends = root["meta/episode_ends"][:]
    source = root["meta/episode_source"][:] if "episode_source" in root["meta"] else np.full(len(episode_ends), -1)
    lengths = np.asarray([end - start for start, end in episode_slices(episode_ends)], dtype=np.int64)
    out = {}
    for source_id, name in ((0, "expert"), (1, "rollout")):
        mask = source == source_id
        out[name] = {
            "episodes": int(mask.sum()),
            "frames": int(lengths[mask].sum()) if mask.any() else 0,
            "episode_length": stats_1d(lengths[mask]) if mask.any() else stats_1d(np.asarray([])),
        }
    total_frames = max(1, out["expert"]["frames"] + out["rollout"]["frames"])
    total_eps = max(1, out["expert"]["episodes"] + out["rollout"]["episodes"])
    out["ratios"] = {
        "expert_episode_fraction": out["expert"]["episodes"] / total_eps,
        "expert_frame_fraction": out["expert"]["frames"] / total_frames,
        "rollout_episode_fraction": out["rollout"]["episodes"] / total_eps,
        "rollout_frame_fraction": out["rollout"]["frames"] / total_frames,
    }
    return out


def summarize_sample_weight(root):
    if "sample_weight" not in root["data"]:
        return {"present": False}
    weights = root["data/sample_weight"][:]
    episode_ends = root["meta/episode_ends"][:]
    source = root["meta/episode_source"][:] if "episode_source" in root["meta"] else np.full(len(episode_ends), -1)
    source_frame_weights = defaultdict(list)
    for ep_idx, (start, end) in enumerate(episode_slices(episode_ends)):
        source_frame_weights[int(source[ep_idx])].append(weights[start:end])
    out = {"present": True, "all": stats_1d(weights)}
    for source_id, name in ((0, "expert"), (1, "rollout")):
        chunks = source_frame_weights.get(source_id, [])
        out[name] = stats_1d(np.concatenate(chunks)) if chunks else stats_1d(np.asarray([]))
    return out


def summarize_alignment(root, max_episodes=50, atol=1e-6):
    states = root["data/state"]
    actions = root["data/action"]
    episode_ends = root["meta/episode_ends"][:]
    checks = []
    for ep_idx, (start, end) in enumerate(episode_slices(episode_ends)):
        if ep_idx >= max_episodes:
            break
        if end - start < 2:
            checks.append({"episode": ep_idx, "frames": int(end - start), "ok": False, "reason": "too_short"})
            continue
        state_next = states[start + 1 : end]
        action_now = actions[start : end - 1]
        max_abs = float(np.max(np.abs(state_next - action_now)))
        checks.append(
            {
                "episode": ep_idx,
                "frames": int(end - start),
                "ok": bool(max_abs <= atol),
                "max_abs_state_next_minus_action": max_abs,
            }
        )
    failed = [row for row in checks if not row["ok"]]
    return {
        "checked_episodes": len(checks),
        "failed_episodes": len(failed),
        "max_abs": max((row.get("max_abs_state_next_minus_action", math.inf) for row in checks), default=math.nan),
        "examples": failed[:5],
    }


def summarize_manifest(task, rollout_dir, rollouts_per_seed):
    manifest_path = rollout_dir / task / "manifest.jsonl"
    rows = list(iter_jsonl(manifest_path))
    success_rows = [row for row in rows if row.get("success")]
    path_rows = [row for row in success_rows if row.get("hdf5_path")]
    missing_paths = []
    for row in path_rows:
        path = rel_or_abs(row["hdf5_path"])
        if not path.is_file():
            missing_paths.append(str(path))

    attempts_by_seed = Counter(int(row["env_seed"]) for row in rows)
    successes_by_seed = Counter(int(row["env_seed"]) for row in success_rows)
    j_hat = {
        str(seed): successes_by_seed[seed] / rollouts_per_seed
        for seed in sorted(attempts_by_seed)
        if rollouts_per_seed > 0
    }
    return {
        "path": str(manifest_path),
        "exists": manifest_path.is_file(),
        "rows": len(rows),
        "success_rows": len(success_rows),
        "success_rows_with_hdf5": len(path_rows),
        "missing_success_hdf5_paths": missing_paths[:20],
        "missing_success_hdf5_count": len(missing_paths),
        "unique_attempt_seeds": len(attempts_by_seed),
        "unique_success_seeds": len(successes_by_seed),
        "attempts_per_seed": stats_1d(np.asarray(list(attempts_by_seed.values()), dtype=np.float32)),
        "successes_per_seed": stats_1d(np.asarray(list(successes_by_seed.values()), dtype=np.float32)),
        "j_hat": stats_1d(np.asarray(list(j_hat.values()), dtype=np.float32)),
    }


def hdf5_quick_check(task, rollout_dir, limit=20):
    try:
        import h5py
    except ImportError:
        return {"skipped": "h5py unavailable"}

    rows = [row for row in iter_jsonl(rollout_dir / task / "manifest.jsonl") if row.get("success") and row.get("hdf5_path")]
    checks = []
    for row in rows[:limit]:
        path = rel_or_abs(row["hdf5_path"])
        item = {"path": str(path), "env_seed": int(row["env_seed"]), "rollout_id": int(row["rollout_id"])}
        try:
            with h5py.File(path, "r") as root:
                vector = root["/joint_action/vector"]
                head = root["/observation/head_camera/rgb"]
                item.update(
                    {
                        "exists": True,
                        "vector_shape": list(vector.shape),
                        "head_camera_frames": int(head.shape[0]),
                        "finite_vector": bool(np.isfinite(vector[:]).all()),
                        "length_match": bool(vector.shape[0] == head.shape[0]),
                    }
                )
        except Exception as exc:
            item.update({"exists": path.is_file(), "error": repr(exc)})
        checks.append(item)
    errors = [row for row in checks if row.get("error") or not row.get("length_match", False)]
    return {"checked": len(checks), "errors": errors[:5], "examples": checks[:5]}


def checkpoint_summary(task, variants, train_seeds, checkpoint_dir, epochs, base_train_seed, expert_data_num):
    out = {}
    base = checkpoint_dir / f"{task}-demo_clean-{expert_data_num}-{base_train_seed}" / "600.ckpt"
    out["base_resume_checkpoint"] = {"path": str(base), "exists": base.is_file(), "bytes": base.stat().st_size if base.is_file() else 0}
    for variant in variants:
        variant_rows = {}
        for seed in train_seeds:
            path = checkpoint_dir / f"{task}-{variant}-{seed}" / f"{epochs}.ckpt"
            variant_rows[str(seed)] = {
                "path": str(path),
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else 0,
            }
        out[variant] = variant_rows
    return out


def log_summary(task, variants, train_seeds, log_dir):
    out = {}
    for variant in variants:
        for seed in train_seeds:
            path = log_dir / f"finetune_{task}_{variant}_seed{seed}.log"
            row = {"path": str(path), "exists": path.is_file()}
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="replace")
                row["loaded_resume_checkpoint"] = bool(re.search(r"Loading model weights from", text))
                row["mentions_dataset"] = f"{task}-{variant}.zarr" in text
                row["last_lines"] = text.splitlines()[-5:]
            out[f"{variant}/seed{seed}"] = row
    return out


def audit_zarr(task, variant, data_dir, expected_action_dim, gripper_indices, max_stat_rows):
    path = data_dir / f"{task}-{variant}.zarr"
    out = {"path": str(path), "exists": path.is_dir()}
    if not path.is_dir():
        return out

    import zarr

    root = zarr.open(str(path), mode="r")
    action = root["data/action"]
    state = root["data/state"]
    head = root["data/head_camera"]
    episode_ends = root["meta/episode_ends"][:]
    out.update(
        {
            "arrays": {
                "state_shape": list(state.shape),
                "action_shape": list(action.shape),
                "head_camera_shape": list(head.shape),
                "episode_count": int(len(episode_ends)),
                "episode_ends_monotonic": bool(np.all(np.diff(episode_ends) > 0)),
                "state_action_same_rows": bool(state.shape[0] == action.shape[0]),
                "state_dim_ok": bool(state.shape[-1] == expected_action_dim),
                "action_dim_ok": bool(action.shape[-1] == expected_action_dim),
            },
            "manifest": read_json(path / "phase1_manifest.json") if (path / "phase1_manifest.json").is_file() else None,
            "sources": summarize_sources(root),
            "sample_weight": summarize_sample_weight(root),
            "alignment": summarize_alignment(root),
            "state_stats_by_dim": array_summary(state, max_rows=max_stat_rows),
            "action_stats_by_dim": array_summary(action, max_rows=max_stat_rows),
        }
    )
    gripper = {}
    for idx in gripper_indices:
        if idx < action.shape[-1]:
            values = action[:, idx]
            gripper[str(idx)] = stats_1d(values)
            gripper[str(idx)]["unique_rounded_3dp"] = int(np.unique(np.round(values[:], 3)).size)
    out["gripper_action_stats"] = gripper
    return out


def make_findings(task_report, expected_action_dim):
    findings = []
    manifest = task_report.get("rollout_manifest", {})
    if not manifest.get("exists"):
        findings.append("rollout manifest is missing")
    elif manifest.get("success_rows", 0) == 0:
        findings.append("no successful rollout episodes found")
    elif manifest.get("missing_success_hdf5_count", 0) > 0:
        findings.append(f"{manifest['missing_success_hdf5_count']} successful rollout hdf5 paths are missing")

    for variant, z in task_report.get("zarr", {}).items():
        if not z.get("exists"):
            findings.append(f"{variant}: zarr dataset is missing")
            continue
        arrays = z["arrays"]
        if not arrays["state_action_same_rows"]:
            findings.append(f"{variant}: state/action row counts differ")
        if not arrays["action_dim_ok"]:
            findings.append(f"{variant}: action dim is {z['arrays']['action_shape'][-1]}, expected {expected_action_dim}")
        if not arrays["state_dim_ok"]:
            findings.append(f"{variant}: state dim is {z['arrays']['state_shape'][-1]}, expected {expected_action_dim}")
        if z["alignment"]["failed_episodes"] > 0:
            findings.append(f"{variant}: state/action offset check failed in {z['alignment']['failed_episodes']} sampled episodes")
        if variant != "expert_only":
            rollout_eps = z["sources"]["rollout"]["episodes"]
            if rollout_eps == 0:
                findings.append(f"{variant}: no rollout episodes in zarr")
            expert_frac = z["sources"]["ratios"]["expert_frame_fraction"]
            if expert_frac < 0.25:
                findings.append(f"{variant}: expert frame fraction is low ({expert_frac:.3f})")

    ckpts = task_report.get("checkpoints", {})
    if not ckpts.get("base_resume_checkpoint", {}).get("exists"):
        findings.append("base resume checkpoint is missing")
    for variant, rows in ckpts.items():
        if variant == "base_resume_checkpoint":
            continue
        for seed, row in rows.items():
            if not row.get("exists"):
                findings.append(f"{variant}/seed{seed}: trained checkpoint is missing")
    return findings


def main():
    parser = argparse.ArgumentParser(description="Audit Phase1 200-demo rollout, zarr, training, and checkpoint integrity.")
    parser.add_argument("--tasks", nargs="+", default=["dump_bin_bigbin"])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(DEFAULT_VARIANTS))
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--rollout-dir", type=Path, default=repo_path("experiments", "phase1", "rollouts_200"))
    parser.add_argument("--data-dir", type=Path, default=repo_path("policy", "DP", "data_phase1_200"))
    parser.add_argument("--checkpoint-dir", type=Path, default=repo_path("policy", "DP", "checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=repo_path("experiments", "phase1", "logs_200"))
    parser.add_argument("--output", type=Path, default=repo_path("experiments", "phase1", "audit_200", "summary.json"))
    parser.add_argument("--expert-data-num", type=int, default=200)
    parser.add_argument("--base-train-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--rollouts-per-seed", type=int, default=8)
    parser.add_argument("--expected-action-dim", type=int, default=14)
    parser.add_argument("--gripper-indices", nargs="+", type=int, default=[6, 13])
    parser.add_argument("--max-stat-rows", type=int, default=200000)
    args = parser.parse_args()
    variants = list(dict.fromkeys(args.variants))

    report = {
        "config": {
            "tasks": args.tasks,
            "variants": variants,
            "train_seeds": args.train_seeds,
            "rollout_dir": str(args.rollout_dir),
            "data_dir": str(args.data_dir),
            "checkpoint_dir": str(args.checkpoint_dir),
            "log_dir": str(args.log_dir),
            "expert_data_num": args.expert_data_num,
            "epochs": args.epochs,
            "rollouts_per_seed": args.rollouts_per_seed,
            "expected_action_dim": args.expected_action_dim,
            "gripper_indices": args.gripper_indices,
        },
        "tasks": {},
    }
    for task in args.tasks:
        task_report = {
            "rollout_manifest": summarize_manifest(task, args.rollout_dir, args.rollouts_per_seed),
            "success_hdf5_quick_check": hdf5_quick_check(task, args.rollout_dir),
            "zarr": {},
            "checkpoints": checkpoint_summary(
                task,
                variants,
                args.train_seeds,
                args.checkpoint_dir,
                args.epochs,
                args.base_train_seed,
                args.expert_data_num,
            ),
            "logs": log_summary(task, variants, args.train_seeds, args.log_dir),
        }
        for variant in variants:
            task_report["zarr"][variant] = audit_zarr(
                task,
                variant,
                args.data_dir,
                args.expected_action_dim,
                args.gripper_indices,
                args.max_stat_rows,
            )
        task_report["findings"] = make_findings(task_report, args.expected_action_dim)
        report["tasks"][task] = task_report

    write_json(args.output, report)
    print(f"Wrote {args.output}")
    for task, task_report in report["tasks"].items():
        findings = task_report["findings"]
        print(f"\n[{task}] findings={len(findings)}")
        for finding in findings:
            print(f"- {finding}")
        if not findings:
            print("- no structural issues found")


if __name__ == "__main__":
    main()
