#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import zarr

from common import (
    VARIANTS,
    add_common_args,
    episode_to_arrays,
    iter_jsonl,
    read_json,
    repo_path,
    write_json,
)


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
                "source": 0,
                "env_seed": -1,
                "rollout_id": -1,
                "j_hat": 1.0,
                "weight": 1.0,
            }
        )
    return episodes


def rollout_success_episodes(task_name, rollout_dir, variant, seed_stats):
    manifest = rollout_dir / task_name / "manifest.jsonl"
    rows = [row for row in iter_jsonl(manifest) if row.get("success") and row.get("hdf5_path")]
    if variant == "seed_balanced":
        picked = {}
        for row in rows:
            picked.setdefault(row["env_seed"], row)
        rows = [picked[k] for k in sorted(picked)]

    episodes = []
    for row in rows:
        env_seed = str(row["env_seed"])
        j_hat = seed_stats[env_seed]["j_hat"]
        weight = difficulty_weight(j_hat) if variant == "difficulty_weighted" else 1.0
        episodes.append(
            {
                "path": repo_path(row["hdf5_path"]),
                "source": 1,
                "env_seed": int(row["env_seed"]),
                "rollout_id": int(row["rollout_id"]),
                "j_hat": float(j_hat),
                "weight": weight,
            }
        )
    return episodes


def build_zarr(episodes, save_dir):
    if save_dir.exists():
        import shutil
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
        path = ep["path"]
        print(f"processing {idx + 1}/{len(episodes)}: {path}")
        head_camera, state, action = episode_to_arrays(path)
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

    if not episode_ends:
        raise RuntimeError("No episodes were added to the dataset.")

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


def main():
    parser = argparse.ArgumentParser(description="Build phase1 zarr datasets for DP self-training variants.")
    add_common_args(parser)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--expert-data-num", type=int, default=50)
    parser.add_argument("--rollout-dir", type=Path, default=repo_path("experiments", "phase1", "rollouts"))
    parser.add_argument("--output-dir", type=Path, default=repo_path("policy", "DP", "data"))
    args = parser.parse_args()

    seed_stats_path = args.rollout_dir / args.task_name / "seed_stats.json"
    seed_stats = read_json(seed_stats_path)
    episodes = expert_episodes(args.task_name, args.task_config, args.expert_data_num)
    episodes.extend(rollout_success_episodes(args.task_name, args.rollout_dir, args.variant, seed_stats))

    suffix = {
        "success": "success",
        "seed_balanced": "seed_balanced",
        "difficulty_weighted": "difficulty_weighted",
    }[args.variant]
    save_dir = args.output_dir / f"{args.task_name}-{suffix}.zarr"
    build_zarr(episodes, save_dir)

    manifest = {
        "task_name": args.task_name,
        "task_config": args.task_config,
        "variant": args.variant,
        "save_dir": str(save_dir),
        "num_episodes": len(episodes),
        "num_expert": args.expert_data_num,
        "num_rollout_success": len(episodes) - args.expert_data_num,
    }
    write_json(save_dir / "phase1_manifest.json", manifest)
    print(f"Wrote {save_dir}")


if __name__ == "__main__":
    main()

