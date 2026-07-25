#!/usr/bin/env python3
"""Build Phase 3 zarr datasets with optional verified failure prefixes.

episode_source labels:
  0 = expert
  1 = rollout success
  2 = verified failure prefix
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path

import numpy as np
import zarr

import sys

# The script directory is already first on sys.path, so this resolves Phase 3's
# common.py. Load Phase 1's same-named module explicitly below to avoid Python's
# module cache silently returning the wrong file.
from common import (  # noqa: E402
    PHASE3,
    SOURCE_EXPERT,
    SOURCE_PREFIX,
    SOURCE_ROLLOUT,
    absorption_config,
    atomic_write_json,
    dataset_path,
)

_PHASE1_COMMON_PATH = Path(__file__).resolve().parents[1] / "phase1" / "common.py"
_PHASE1_SPEC = importlib.util.spec_from_file_location("phase1_common_for_phase3", _PHASE1_COMMON_PATH)
if _PHASE1_SPEC is None or _PHASE1_SPEC.loader is None:
    raise ImportError(f"Cannot load Phase 1 helpers from {_PHASE1_COMMON_PATH}")
_PHASE1_COMMON = importlib.util.module_from_spec(_PHASE1_SPEC)
_PHASE1_SPEC.loader.exec_module(_PHASE1_COMMON)

add_common_args = _PHASE1_COMMON.add_common_args
episode_to_arrays = _PHASE1_COMMON.episode_to_arrays
iter_jsonl = _PHASE1_COMMON.iter_jsonl
read_json = _PHASE1_COMMON.read_json
repo_path = _PHASE1_COMMON.repo_path
write_json = _PHASE1_COMMON.write_json


def difficulty_weight(j_hat, eps=0.05, alpha=0.5, w_min=1.0, w_max=4.0):
    return float(np.clip(1.0 / ((float(j_hat) + eps) ** alpha), w_min, w_max))


def expert_episodes(task_name, task_config, expert_data_num):
    base = repo_path("data", task_name, task_config, "data")
    episodes = []
    for idx in range(expert_data_num):
        path = base / f"episode{idx}.hdf5"
        episodes.append(
            {
                "path": path,
                "source": SOURCE_EXPERT,
                "env_seed": -1,
                "rollout_id": -1,
                "j_hat": 1.0,
                "weight": 1.0,
            }
        )
    return episodes


def _success_rows(manifest_path: Path) -> list[dict]:
    return [row for row in iter_jsonl(manifest_path) if row.get("success") and row.get("hdf5_path")]


def _failure_rows(manifest_path: Path) -> list[dict]:
    return [
        row
        for row in iter_jsonl(manifest_path)
        if not row.get("success") and row.get("failure_hdf5_path")
    ]


def select_success_rows(rows: list[dict], diversity_mode: str) -> list[dict]:
    if diversity_mode == "seed_balanced":
        picked = {}
        for row in rows:
            picked.setdefault(row["env_seed"], row)
        return [picked[k] for k in sorted(picked)]
    return list(rows)


def extract_prefix_episode(
    path: Path,
    *,
    env_seed: int,
    rollout_id: int,
    j_hat: float,
    prefix_fraction: float,
    min_prefix_steps: int,
    extraction_mode: str,
) -> dict | None:
    head_camera, state, action = episode_to_arrays(path)
    total = len(state)
    if total < min_prefix_steps + 1:
        return None
    prefix_steps = max(min_prefix_steps, int(total * prefix_fraction))
    prefix_steps = min(prefix_steps, total - 1)
    return {
        "head_camera": head_camera[:prefix_steps],
        "state": state[:prefix_steps],
        "action": action[:prefix_steps],
        "source": SOURCE_PREFIX,
        "env_seed": env_seed,
        "rollout_id": rollout_id,
        "j_hat": j_hat,
        "weight": 1.0,
        "total_steps": total,
        "prefix_steps": prefix_steps,
        "prefix_fraction": prefix_fraction,
        "extraction_mode": extraction_mode,
        "path": str(path),
    }


def build_zarr(episodes, save_dir: Path):
    if save_dir.exists():
        shutil.rmtree(save_dir)

    zarr_root = zarr.group(str(save_dir))
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    head_camera_arrays = []
    state_arrays = []
    action_arrays = []
    sample_weight_arrays = []
    episode_ends = []
    episode_env_seed = []
    episode_rollout_id = []
    episode_source = []
    episode_seed_j_hat = []
    episode_sample_weight = []

    total_count = 0
    for idx, ep in enumerate(episodes):
        if "head_camera" in ep:
            head_camera, state, action = ep["head_camera"], ep["state"], ep["action"]
        else:
            head_camera, state, action = episode_to_arrays(ep["path"])
        if len(state) == 0:
            continue
        head_camera_arrays.append(head_camera)
        state_arrays.append(state)
        action_arrays.append(action)
        sample_weight_arrays.append(np.full((state.shape[0],), ep["weight"], dtype=np.float32))

        total_count += state.shape[0]
        episode_ends.append(total_count)
        episode_env_seed.append(ep["env_seed"])
        episode_rollout_id.append(ep["rollout_id"])
        episode_source.append(ep["source"])
        episode_seed_j_hat.append(ep["j_hat"])
        episode_sample_weight.append(ep["weight"])

    head_camera_arrays = np.concatenate(head_camera_arrays, axis=0)
    state_arrays = np.concatenate(state_arrays, axis=0).astype(np.float32)
    action_arrays = np.concatenate(action_arrays, axis=0).astype(np.float32)
    sample_weight_arrays = np.concatenate(sample_weight_arrays, axis=0).astype(np.float32)

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    zarr_data.create_dataset(
        "head_camera",
        data=head_camera_arrays,
        chunks=(100, *head_camera_arrays.shape[1:]),
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "state",
        data=state_arrays,
        chunks=(100, state_arrays.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "action",
        data=action_arrays,
        chunks=(100, action_arrays.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "sample_weight",
        data=sample_weight_arrays,
        chunks=(100,),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_meta.create_dataset("episode_ends", data=np.asarray(episode_ends), dtype="int64", overwrite=True)
    zarr_meta.create_dataset("episode_env_seed", data=np.asarray(episode_env_seed), dtype="int64", overwrite=True)
    zarr_meta.create_dataset("episode_rollout_id", data=np.asarray(episode_rollout_id), dtype="int64", overwrite=True)
    zarr_meta.create_dataset("episode_source", data=np.asarray(episode_source), dtype="int64", overwrite=True)
    zarr_meta.create_dataset("episode_seed_j_hat", data=np.asarray(episode_seed_j_hat), dtype="float32", overwrite=True)
    zarr_meta.create_dataset(
        "episode_sample_weight",
        data=np.asarray(episode_sample_weight),
        dtype="float32",
        overwrite=True,
    )


def build_dataset_for_absorption(
    task: str,
    task_config: str,
    main_id: str,
    *,
    rollout_dir: Path,
    expert_data_num: int = 200,
    prefix_fraction: float = 0.7,
    min_prefix_steps: int = 5,
    extraction_mode: str = "fixed_fraction",
) -> tuple[Path, list[dict]]:
    cfg = absorption_config(main_id)
    seed_stats = read_json(rollout_dir / task / "seed_stats.json")
    manifest = rollout_dir / task / "manifest.jsonl"

    episodes = expert_episodes(task, task_config, expert_data_num)
    prefix_candidates = []

    if cfg["dataset_variant"] != "expert_only":
        success_rows = select_success_rows(_success_rows(manifest), cfg["diversity_mode"])
        for row in success_rows:
            env_seed = str(row["env_seed"])
            j_hat = seed_stats[env_seed]["j_hat"]
            weight = difficulty_weight(j_hat) if cfg["diversity_mode"] == "seed_balanced" else 1.0
            episodes.append(
                {
                    "path": repo_path(row["hdf5_path"]),
                    "source": SOURCE_ROLLOUT,
                    "env_seed": int(row["env_seed"]),
                    "rollout_id": int(row["rollout_id"]),
                    "j_hat": float(j_hat),
                    "weight": weight,
                }
            )

        if cfg["use_failure_prefix"]:
            for row in _failure_rows(manifest):
                env_seed = str(row["env_seed"])
                j_hat = seed_stats[env_seed]["j_hat"]
                prefix_ep = extract_prefix_episode(
                    repo_path(row["failure_hdf5_path"]),
                    env_seed=int(row["env_seed"]),
                    rollout_id=int(row["rollout_id"]),
                    j_hat=float(j_hat),
                    prefix_fraction=prefix_fraction,
                    min_prefix_steps=min_prefix_steps,
                    extraction_mode=extraction_mode,
                )
                if prefix_ep is None:
                    continue
                prefix_candidates.append(
                    {
                        "task": task,
                        "env_seed": prefix_ep["env_seed"],
                        "rollout_id": prefix_ep["rollout_id"],
                        "hdf5_path": prefix_ep["path"],
                        "total_steps": prefix_ep["total_steps"],
                        "prefix_steps": prefix_ep["prefix_steps"],
                        "prefix_fraction": prefix_ep["prefix_fraction"],
                        "extraction_mode": extraction_mode,
                        "j_hat": prefix_ep["j_hat"],
                    }
                )
                episodes.append(prefix_ep)

    suffix = cfg["dataset_variant"]
    save_dir = dataset_path(task, suffix)
    build_zarr(episodes, save_dir)
    return save_dir, prefix_candidates


def main():
    parser = argparse.ArgumentParser(description="Build Phase 3 CPST zarr datasets.")
    add_common_args(parser)
    parser.add_argument("--main-id", required=True, choices=["A0", "A1", "A2", "A3", "A4"])
    parser.add_argument("--rollout-dir", type=Path, default=repo_path("experiments", "phase1", "rollouts_200"))
    parser.add_argument("--expert-data-num", type=int, default=200)
    parser.add_argument("--prefix-fraction", type=float, default=0.7)
    parser.add_argument("--min-prefix-steps", type=int, default=5)
    parser.add_argument("--extraction-mode", default="fixed_fraction")
    parser.add_argument("--write-candidates", action="store_true")
    args = parser.parse_args()

    save_dir, prefix_candidates = build_dataset_for_absorption(
        args.task_name,
        args.task_config,
        args.main_id,
        rollout_dir=args.rollout_dir,
        expert_data_num=args.expert_data_num,
        prefix_fraction=args.prefix_fraction,
        min_prefix_steps=args.min_prefix_steps,
        extraction_mode=args.extraction_mode,
    )

    manifest = {
        "task_name": args.task_name,
        "main_id": args.main_id,
        "absorption_id": absorption_config(args.main_id)["absorption_id"],
        "save_dir": str(save_dir),
        "num_prefix_candidates": len(prefix_candidates),
    }
    write_json(save_dir / "phase3_manifest.json", manifest)
    print(f"Wrote {save_dir}")

    if args.write_candidates and prefix_candidates:
        out = PHASE3 / "prefix_candidates" / f"{args.task_name}.json"
        atomic_write_json(out, {"task": args.task_name, "candidates": prefix_candidates})
        print(f"Wrote prefix candidates: {out}")


if __name__ == "__main__":
    main()
