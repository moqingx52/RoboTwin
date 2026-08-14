#!/usr/bin/env python3
"""Build an independent anchor replay zarr with capability-group preservation labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[2]

from experiments.brace.preservation_groups import (
    PRESERVATION_BASE_SOLVED,
    PRESERVATION_BOUNDARY,
    PRESERVATION_HARD_MONITOR,
    SOURCE_ROLLOUT,
)
from experiments.brace.replay_audit import collect_candidates, read_json, repo_path


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_rgb(value) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(value, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode head-camera frame")
    return np.moveaxis(image, -1, 0)


def extract_mid_episode_window(hdf5_path: Path, *, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(hdf5_path, "r") as root:
        vector = np.asarray(root["/joint_action/vector"], dtype=np.float32)
        frames = root["/observation/head_camera/rgb"]
        if len(vector) < horizon + 1:
            raise ValueError(f"trajectory too short in {hdf5_path}")
        mid = max(0, (len(vector) - horizon - 1) // 2)
        indices = np.arange(mid, mid + horizon)
        state = vector[indices]
        action = vector[np.clip(indices + 1, 1, len(vector) - 1)].copy()
        head_camera = np.stack([decode_rgb(frames[int(idx)]) for idx in indices])
    return head_camera, state, action


def classify_env_seeds(
    candidates,
    *,
    hard_seeds: set[int],
    min_rollouts_per_seed: int,
    min_base_success_rate: float,
) -> dict[int, str]:
    by_seed: dict[int, list[bool]] = defaultdict(list)
    for candidate in candidates:
        by_seed[int(candidate.env_seed)].append(bool(candidate.success))

    labels: dict[int, str] = {}
    for env_seed, outcomes in sorted(by_seed.items()):
        if env_seed in hard_seeds:
            labels[env_seed] = "hard_monitor"
            continue
        if len(outcomes) < min_rollouts_per_seed:
            continue
        success_rate = sum(outcomes) / len(outcomes)
        if success_rate >= min_base_success_rate:
            labels[env_seed] = "base_solved"
        elif 0.0 < success_rate < min_base_success_rate:
            labels[env_seed] = "boundary"
    return labels


def select_representative_candidate(candidates, env_seed: int, *, prefer_success: bool | None):
    pool = [item for item in candidates if int(item.env_seed) == env_seed]
    if not pool:
        return None
    if prefer_success is True:
        pool = [item for item in pool if item.success] or pool
    elif prefer_success is False:
        pool = [item for item in pool if not item.success] or pool
    return sorted(pool, key=lambda item: int(item.rollout_id))[0]


def build_anchor_replay_set(
    *,
    task: str,
    rollout_dir: Path,
    output_path: Path,
    hard_seeds_file: Path | None,
    horizon: int = 8,
    min_rollouts_per_seed: int = 2,
    min_base_success_rate: float = 1.0,
    min_env_seeds_per_group: int = 2,
    max_episodes_per_group: int = 8,
) -> dict:
    candidates, errors = collect_candidates(task, rollout_dir)
    if errors:
        raise RuntimeError(f"failed to collect rollout candidates: {errors[:3]}")

    hard_seeds: set[int] = set()
    if hard_seeds_file and hard_seeds_file.is_file():
        payload = read_json(hard_seeds_file)
        hard_seeds = {int(seed) for seed in payload.get("hard_seeds", [])}

    seed_labels = classify_env_seeds(
        candidates,
        hard_seeds=hard_seeds,
        min_rollouts_per_seed=min_rollouts_per_seed,
        min_base_success_rate=min_base_success_rate,
    )
    group_map = {
        "base_solved": PRESERVATION_BASE_SOLVED,
        "boundary": PRESERVATION_BOUNDARY,
        "hard_monitor": PRESERVATION_HARD_MONITOR,
    }
    episodes: list[dict] = []
    for env_seed, label in seed_labels.items():
        prefer_success = True if label == "base_solved" else False if label == "boundary" else None
        candidate = select_representative_candidate(candidates, env_seed, prefer_success=prefer_success)
        if candidate is None:
            continue
        episodes.append(
            {
                "env_seed": env_seed,
                "rollout_id": int(candidate.rollout_id),
                "preservation_group": label,
                "preservation_group_id": group_map[label],
                "base_outcome": bool(candidate.success),
                "hdf5_path": str(candidate.path),
            }
        )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in episodes:
        grouped[row["preservation_group"]].append(row)
    for group_name, rows in grouped.items():
        if group_name != "hard_monitor" and len({row["env_seed"] for row in rows}) < min_env_seeds_per_group:
            raise ValueError(
                f"group {group_name} has fewer than {min_env_seeds_per_group} unique env seeds: "
                f"{sorted({row['env_seed'] for row in rows})}"
            )
        grouped[group_name] = rows[:max_episodes_per_group]

    selected = [row for group_rows in grouped.values() for row in group_rows]
    if not selected:
        raise ValueError("no anchor replay episodes selected")
    for required_group in ("base_solved", "boundary"):
        if required_group not in grouped or not grouped[required_group]:
            raise ValueError(f"anchor replay set missing required preservation group: {required_group}")

    chunks = [extract_mid_episode_window(Path(row["hdf5_path"]), horizon=horizon) for row in selected]
    total_frames = len(chunks) * horizon

    tmp = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    if tmp.exists():
        raise FileExistsError(tmp)
    if output_path.exists():
        manifest = output_path / "brace_anchor_manifest.json"
        if manifest.is_file():
            return json.loads(manifest.read_text(encoding="utf-8"))
        raise FileExistsError(f"refusing to overwrite anchor replay set: {output_path}")

    root = zarr.group(str(tmp))
    data = root.create_group("data")
    meta = root.create_group("meta")
    try:
        sample = chunks[0]
        for key, array in zip(("head_camera", "state", "action"), sample):
            shape = (total_frames, *array.shape[1:])
            data.create_dataset(key, shape=shape, chunks=(min(horizon, total_frames), *array.shape[1:]), dtype=array.dtype)
        offset = 0
        for images, states, actions in chunks:
            stop = offset + horizon
            data["head_camera"][offset:stop] = images
            data["state"][offset:stop] = states
            data["action"][offset:stop] = actions
            offset = stop

        ends = horizon * np.arange(1, len(chunks) + 1, dtype=np.int64)
        meta.create_dataset("episode_ends", data=ends, dtype="int64")
        meta.create_dataset("episode_source", data=np.full(len(chunks), SOURCE_ROLLOUT, dtype=np.int64), dtype="int64")
        meta.create_dataset(
            "episode_preservation_group",
            data=np.asarray([row["preservation_group_id"] for row in selected], dtype=np.int64),
            dtype="int64",
        )
        meta.create_dataset(
            "episode_env_seed",
            data=np.asarray([row["env_seed"] for row in selected], dtype=np.int64),
            dtype="int64",
        )

        manifest_rows = [
            {
                "env_seed": row["env_seed"],
                "rollout_id": row["rollout_id"],
                "preservation_group": row["preservation_group"],
                "base_outcome": row["base_outcome"],
                "hdf5_path": row["hdf5_path"],
                "hdf5_sha256": sha256_file(Path(row["hdf5_path"])),
            }
            for row in selected
        ]
        manifest = {
            "schema_version": 1,
            "task": task,
            "rollout_dir": str(rollout_dir.resolve()),
            "horizon": horizon,
            "episodes": manifest_rows,
            "group_counts": {group: len(rows) for group, rows in grouped.items()},
            "unique_env_seeds": sorted({row["env_seed"] for row in selected}),
            "manifest_sha256": sha256_bytes(
                json.dumps(manifest_rows, sort_keys=True).encode("utf-8")
            ),
        }
        (tmp / "brace_anchor_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, output_path)
        return manifest
    except Exception:
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hard-seeds-file", type=Path)
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()
    payload = build_anchor_replay_set(
        task=args.task,
        rollout_dir=repo_path(args.rollout_dir),
        output_path=repo_path(args.output),
        hard_seeds_file=repo_path(args.hard_seeds_file) if args.hard_seeds_file else None,
        horizon=args.horizon,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
