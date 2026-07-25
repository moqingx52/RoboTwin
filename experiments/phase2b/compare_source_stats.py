#!/usr/bin/env python3
"""Compare expert vs rollout source statistics for Phase 2b diagnose."""

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DEFAULT_TASKS, atomic_write_json, dataset_path, repo_path


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
            "p05": math.nan,
            "p50": math.nan,
            "p95": math.nan,
            "p99": math.nan,
            "max": math.nan,
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p01": pct(values, 1),
        "p05": pct(values, 5),
        "p50": pct(values, 50),
        "p95": pct(values, 95),
        "p99": pct(values, 99),
        "max": float(values.max()),
    }


def episode_slices(episode_ends):
    start = 0
    for end in episode_ends:
        end = int(end)
        yield start, end
        start = end


def per_dim_summary(arr, source_mask):
    out = []
    for dim in range(arr.shape[1]):
        values = arr[source_mask, dim]
        row = stats_1d(values)
        row["dim"] = dim
        out.append(row)
    return out


def action_delta_stats(actions, episode_ends, episode_sources, source_id):
    deltas = []
    for ep_idx, (start, end) in enumerate(episode_slices(episode_ends)):
        if int(episode_sources[ep_idx]) != source_id or end - start < 2:
            continue
        ep_actions = actions[start:end]
        deltas.append(np.linalg.norm(np.diff(ep_actions, axis=0), axis=1))
    if not deltas:
        return stats_1d(np.asarray([]))
    return stats_1d(np.concatenate(deltas))


def normalizer_from_arrays(states, actions):
    from diffusion_policy.model.common.normalizer import LinearNormalizer

    normalizer = LinearNormalizer()
    normalizer.fit(
        data={"action": actions, "agent_pos": states},
        last_n_dims=1,
        mode="limits",
    )
    return normalizer


def normalizer_summary(normalizer):
    out = {}
    for key in ("action", "agent_pos"):
        field = normalizer[key]
        out[key] = {
            "scale": field.params_dict["scale"].detach().cpu().numpy().tolist(),
            "offset": field.params_dict["offset"].detach().cpu().numpy().tolist(),
        }
    return out


def summarize_task(task, gripper_indices):
    zarr_path = dataset_path(task, "success")
    root = zarr.open(str(zarr_path), mode="r")
    states = np.asarray(root["data/state"][:], dtype=np.float32)
    actions = np.asarray(root["data/action"][:], dtype=np.float32)
    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    episode_sources = np.asarray(root["meta/episode_source"][:], dtype=np.int64)

    frame_sources = np.repeat(
        episode_sources,
        np.diff(np.concatenate(([0], episode_ends))),
    )
    expert_mask = frame_sources == 0
    rollout_mask = frame_sources == 1

    episode_lengths = defaultdict(list)
    for ep_idx, (start, end) in enumerate(episode_slices(episode_ends)):
        episode_lengths[int(episode_sources[ep_idx])].append(end - start)

    gripper = {}
    for idx in gripper_indices:
        gripper[str(idx)] = {
            "expert": stats_1d(actions[expert_mask, idx]),
            "rollout": stats_1d(actions[rollout_mask, idx]),
        }

    expert_normalizer = normalizer_from_arrays(states[expert_mask], actions[expert_mask])
    mixed_normalizer = normalizer_from_arrays(states, actions)

    return {
        "task": task,
        "zarr_path": str(zarr_path),
        "frame_counts": {
            "expert": int(expert_mask.sum()),
            "rollout": int(rollout_mask.sum()),
        },
        "state": {
            "expert": per_dim_summary(states, expert_mask),
            "rollout": per_dim_summary(states, rollout_mask),
        },
        "action": {
            "expert": per_dim_summary(actions, expert_mask),
            "rollout": per_dim_summary(actions, rollout_mask),
        },
        "action_delta_l2": {
            "expert": action_delta_stats(actions, episode_ends, episode_sources, 0),
            "rollout": action_delta_stats(actions, episode_ends, episode_sources, 1),
        },
        "episode_length": {
            "expert": stats_1d(np.asarray(episode_lengths.get(0, []), dtype=np.int64)),
            "rollout": stats_1d(np.asarray(episode_lengths.get(1, []), dtype=np.int64)),
        },
        "gripper_action_stats": gripper,
        "normalizer": {
            "expert_only_fit": normalizer_summary(expert_normalizer),
            "mixed_fit": normalizer_summary(mixed_normalizer),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Compare expert vs rollout dataset statistics.")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--gripper-indices", nargs="+", type=int, default=[6, 13])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "diagnose" / "source_stats.json",
    )
    args = parser.parse_args()

    import sys

    sys.path.append(str(repo_path("policy", "DP")))

    summary = {"tasks": {}}
    for task in args.tasks:
        summary["tasks"][task] = summarize_task(task, args.gripper_indices)
    atomic_write_json(args.output, summary)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
