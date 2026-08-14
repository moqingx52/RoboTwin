#!/usr/bin/env python3
"""Build the BRACE-RW return-weighted zarr dataset from E0 outputs.

Each E0 branch point s contributed A sampled action chunks a_1..a_A (drawn
from the frozen policy pi_0 at the restored state) with R executed
continuations each. This builder turns every (point, action) pair into one
training episode whose action window is the sampled chunk and whose per-frame
`data/sample_weight` is the BRACE-RW weight

    w_a = (1 + c * G_bar_a) / (1 + c * V_hat(s)),

with G_bar_a = successes_a / R (unbiased for Q0(s,a)) and
V_hat(s) = mean_a G_bar_a. Expert episodes are appended with weight 1.0
(the N1' replay component). Setting c=0 reproduces N1' exactly (all weights
1), so the same builder emits both arms.

The training hook already exists: RobotImageDataset picks up
`data/sample_weight` and `aggregate_training_loss` (robotworkspace.py) computes
sum(w_i * loss_i) / sum(w_i) in pooled mode.

Observation frames/states come from the traced success rollout HDF5 at the
branch boundary — identical to the state the chunks were sampled at.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.build_screen_dataset import (
    _copy_array,
    decode_rgb,
    read_jsonl,
    sha256,
)
from experiments.brace.preservation_groups import PRESERVATION_NONE, SOURCE_EXPERT, SOURCE_ROLLOUT
from experiments.brace.replay_audit import collect_candidates, repo_path


def load_e0_points(e0_dir: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    """Group checks.jsonl rollouts into per-point action outcome tables."""
    rows = read_jsonl(e0_dir / "checks.jsonl")
    grouped: dict[tuple[str, int, int], dict[str, Any]] = {}
    outcomes: dict[tuple[str, int, int], dict[int, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (str(row["task"]), int(row["env_seed"]), int(row["snapshot_id"]))
        outcomes[key][int(row["action_index"])].append(bool(row["success"]))
        grouped[key] = {
            "task": row["task"],
            "env_seed": int(row["env_seed"]),
            "snapshot_id": int(row["snapshot_id"]),
            "branch_chunk_index": int(row["branch_chunk_index"]),
            "point_type": row.get("point_type"),
            "success_rollout_id": row.get("success_rollout_id"),
        }
    for key, point in grouped.items():
        table = outcomes[key]
        g_bar = {a: sum(flags) / len(flags) for a, flags in sorted(table.items())}
        point["g_bar"] = g_bar
        point["replicates"] = {a: len(flags) for a, flags in sorted(table.items())}
        point["v_hat"] = sum(g_bar.values()) / len(g_bar)
    return grouped


def load_point_chunks(e0_dir: Path, point: dict[str, Any]) -> np.ndarray:
    npz_path = e0_dir / "chunks" / f"{point['task']}_seed{point['env_seed']}_snap{point['snapshot_id']}.npz"
    with np.load(npz_path) as payload:
        return np.asarray(payload["actions"], dtype=np.float32)


def extract_boundary_window(
    hdf5_path: Path,
    branch_chunk_index: int,
    *,
    horizon: int,
    n_obs_steps: int,
    n_action_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read (head_camera, state, action) around the branch boundary from the trace."""
    import h5py

    with h5py.File(hdf5_path, "r") as root:
        vector = np.asarray(root["/joint_action/vector"], dtype=np.float32)
        frames = root["/observation/head_camera/rgb"]
        boundary = int(branch_chunk_index) * n_action_steps
        sequence_start = boundary - (n_obs_steps - 1)
        max_state_index = min(len(vector) - 2, len(frames) - 2)
        indices = np.clip(np.arange(sequence_start, sequence_start + horizon), 0, max_state_index)
        state = vector[indices]
        action = vector[np.clip(indices + 1, 1, len(vector) - 1)].copy()
        head_camera = np.stack([decode_rgb(frames[int(idx)]) for idx in indices])
    return head_camera, state, action


def build_rw_dataset(
    expert_path: Path,
    e0_dir: Path,
    output_path: Path,
    *,
    rollout_dir: Path,
    c: float,
    horizon: int,
    n_obs_steps: int,
    n_action_steps: int,
    min_replicates: int = 1,
) -> dict:
    points = load_e0_points(e0_dir)
    if not points:
        raise ValueError(f"no points found in {e0_dir / 'checks.jsonl'}")

    trace_paths: dict[tuple[str, int], Path] = {}
    for task in {point["task"] for point in points.values()}:
        candidates, _ = collect_candidates(task, rollout_dir)
        for candidate in candidates:
            trace_paths[(task, candidate.rollout_id)] = candidate.path

    episodes: list[dict[str, Any]] = []
    for key in sorted(points):
        point = points[key]
        chunks = load_point_chunks(e0_dir, point)
        trace_path = trace_paths.get((point["task"], point["success_rollout_id"]))
        if trace_path is None:
            raise FileNotFoundError(
                f"success trace rollout_id={point['success_rollout_id']} for {key} not found under {rollout_dir}"
            )
        head_camera, state, base_action = extract_boundary_window(
            trace_path,
            point["branch_chunk_index"],
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
        )
        action_start = n_obs_steps - 1
        for action_index, g_bar in point["g_bar"].items():
            if point["replicates"][action_index] < min_replicates:
                continue
            chunk = chunks[action_index]
            if chunk.shape[0] < n_action_steps:
                raise ValueError(f"chunk too short at {key} action {action_index}: {chunk.shape}")
            action = base_action.copy()
            action[action_start : action_start + n_action_steps] = chunk[:n_action_steps]
            weight = (1.0 + c * g_bar) / (1.0 + c * point["v_hat"])
            episodes.append(
                {
                    "head_camera": head_camera,
                    "state": state,
                    "action": action,
                    "weight": float(weight),
                    "env_seed": point["env_seed"],
                    "g_bar": float(g_bar),
                    "v_hat": float(point["v_hat"]),
                    "meta": {
                        "task": point["task"],
                        "env_seed": point["env_seed"],
                        "snapshot_id": point["snapshot_id"],
                        "point_type": point["point_type"],
                        "action_index": action_index,
                        "g_bar": float(g_bar),
                        "v_hat": float(point["v_hat"]),
                        "weight": float(weight),
                    },
                }
            )

    source = zarr.open(str(expert_path), mode="r")
    expert_frames = int(source["meta/episode_ends"][-1])
    expert_episodes = len(source["meta/episode_ends"])
    total_frames = expert_frames + len(episodes) * horizon

    checks_sha = sha256(e0_dir / "checks.jsonl")
    tmp = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    if tmp.exists():
        raise FileExistsError(tmp)
    if output_path.exists():
        metadata = output_path / "brace_rw_manifest.json"
        if metadata.is_file():
            existing = json.loads(metadata.read_text())
            if existing.get("e0_checks_sha256") == checks_sha and existing.get("c") == c:
                return existing
        raise FileExistsError(f"refusing to overwrite dataset: {output_path}")

    root = zarr.group(str(tmp))
    data = root.create_group("data")
    meta = root.create_group("meta")

    for zarr_key in ("head_camera", "state", "action"):
        src = source[f"data/{zarr_key}"]
        shape = (total_frames, *src.shape[1:])
        chunks_shape = src.chunks or (min(100, total_frames), *src.shape[1:])
        dst = data.create_dataset(
            zarr_key,
            shape=shape,
            chunks=chunks_shape,
            dtype=src.dtype,
            compressor=getattr(src, "compressor", None),
            overwrite=False,
        )
        _copy_array(src, dst)

    sample_weight = data.create_dataset(
        "sample_weight", shape=(total_frames,), chunks=(min(4096, total_frames),), dtype="float32"
    )
    sample_weight[:expert_frames] = 1.0

    for episode_idx, episode in enumerate(episodes):
        start = expert_frames + episode_idx * horizon
        stop = start + horizon
        data["head_camera"][start:stop] = episode["head_camera"]
        data["state"][start:stop] = episode["state"]
        data["action"][start:stop] = episode["action"]
        sample_weight[start:stop] = episode["weight"]

    expert_ends = np.asarray(source["meta/episode_ends"][:], dtype=np.int64)
    new_ends = expert_frames + horizon * np.arange(1, len(episodes) + 1, dtype=np.int64)
    meta.create_dataset("episode_ends", data=np.concatenate((expert_ends, new_ends)), dtype="int64")
    source_labels = np.concatenate(
        (
            np.full(expert_episodes, SOURCE_EXPERT, dtype=np.int64),
            np.full(len(episodes), SOURCE_ROLLOUT, dtype=np.int64),
        )
    )
    meta.create_dataset("episode_source", data=source_labels, dtype="int64")
    preservation_labels = np.full(expert_episodes + len(episodes), PRESERVATION_NONE, dtype=np.int64)
    meta.create_dataset("episode_preservation_group", data=preservation_labels, dtype="int64")
    env_seed = np.concatenate(
        (
            np.full(expert_episodes, -1, dtype=np.int64),
            np.asarray([episode["env_seed"] for episode in episodes], dtype=np.int64),
        )
    )
    meta.create_dataset("episode_env_seed", data=env_seed, dtype="int64")
    episode_weight = np.concatenate(
        (
            np.full(expert_episodes, 1.0, dtype=np.float64),
            np.asarray([episode["weight"] for episode in episodes], dtype=np.float64),
        )
    )
    meta.create_dataset("episode_sample_weight", data=episode_weight, dtype="float64")

    weights = [episode["weight"] for episode in episodes]
    payload = {
        "schema_version": 1,
        "arm": "N1_prime" if c == 0.0 else "B_rw",
        "c": c,
        "expert_path": str(expert_path.resolve()),
        "e0_dir": str(e0_dir.resolve()),
        "e0_checks_sha256": checks_sha,
        "expert_episodes": expert_episodes,
        "chunk_episodes": len(episodes),
        "points": len(points),
        "horizon": horizon,
        "n_obs_steps": n_obs_steps,
        "n_action_steps": n_action_steps,
        "weight_stats": {
            "min": min(weights) if weights else None,
            "max": max(weights) if weights else None,
            "mean": float(np.mean(weights)) if weights else None,
            "theoretical_max": 1.0 + c,
        },
        "episodes": [episode["meta"] for episode in episodes],
    }
    (tmp / "brace_rw_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, output_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-zarr", type=Path, required=True)
    parser.add_argument("--e0-dir", type=Path, required=True, help="collect_e0_variance.py output dir")
    parser.add_argument("--rollout-dir", type=Path, required=True, help="traced rollouts (for boundary obs)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--c", type=float, required=True, help="weight coefficient; 0 -> N1' control arm")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--n-obs-steps", type=int, default=3)
    parser.add_argument("--n-action-steps", type=int, default=6)
    args = parser.parse_args()
    payload = build_rw_dataset(
        repo_path(args.expert_zarr),
        repo_path(args.e0_dir),
        repo_path(args.output),
        rollout_dir=repo_path(args.rollout_dir),
        c=args.c,
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
    )
    summary = {k: v for k, v in payload.items() if k != "episodes"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
