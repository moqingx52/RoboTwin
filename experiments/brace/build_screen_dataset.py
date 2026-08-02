#!/usr/bin/env python3
"""Build a DP zarr dataset from expert episodes plus BRACE chunk manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_hdf5(record: dict, traced_root: Path | None) -> Path:
    raw = Path(record["hdf5_path"])
    candidates = [raw, REPO_ROOT / raw]
    parts = raw.parts
    if "experiments" in parts:
        candidates.append(REPO_ROOT.joinpath(*parts[parts.index("experiments") :]))
    if traced_root is not None:
        candidates.append(traced_root / record["task"] / "successes" / raw.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot resolve HDF5 for {raw}; tried: {candidates}")


def decode_rgb(value) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(value, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode head-camera frame")
    return np.moveaxis(image, -1, 0)


def extract_chunk(record: dict, hdf5_path: Path, *, horizon: int, n_obs_steps: int, n_action_steps: int):
    with h5py.File(hdf5_path, "r") as root:
        vector = np.asarray(root["/joint_action/vector"], dtype=np.float32)
        frames = root["/observation/head_camera/rgb"]
        chunk_indices = np.asarray(root["/policy_chunks/chunk_index"], dtype=np.int64)
        matches = np.flatnonzero(chunk_indices == int(record["branch_chunk_index"]))
        if len(matches) != 1:
            raise ValueError(f"expected one policy chunk in {hdf5_path}, found {len(matches)}")
        chunk_action = np.asarray(root["/policy_chunks/action"][int(matches[0])], dtype=np.float32)
        if chunk_action.shape != (n_action_steps, vector.shape[1]):
            raise ValueError(
                f"chunk action shape {chunk_action.shape} != {(n_action_steps, vector.shape[1])} in {hdf5_path}"
            )

        boundary = int(record["branch_chunk_index"]) * n_action_steps
        sequence_start = boundary - (n_obs_steps - 1)
        max_state_index = min(len(vector) - 2, len(frames) - 2)
        indices = np.clip(np.arange(sequence_start, sequence_start + horizon), 0, max_state_index)
        state = vector[indices]
        action = vector[np.clip(indices + 1, 1, len(vector) - 1)].copy()
        action_start = n_obs_steps - 1
        action[action_start : action_start + n_action_steps] = chunk_action
        head_camera = np.stack([decode_rgb(frames[int(idx)]) for idx in indices])
    return head_camera, state, action


def _copy_array(source, target, offset=0, block_size=256):
    for start in range(0, len(source), block_size):
        stop = min(len(source), start + block_size)
        target[offset + start : offset + stop] = source[start:stop]


def build_dataset(
    expert_path: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    traced_root: Path | None,
    horizon: int,
    n_obs_steps: int,
    n_action_steps: int,
) -> dict:
    rows = read_jsonl(manifest_path)
    if not rows:
        raise ValueError(f"empty manifest: {manifest_path}")
    source = zarr.open(str(expert_path), mode="r")
    expert_frames = int(source["meta/episode_ends"][-1])
    expert_episodes = len(source["meta/episode_ends"])
    chunks = [
        extract_chunk(
            row,
            resolve_hdf5(row, traced_root),
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
        )
        for row in rows
    ]
    total_frames = expert_frames + len(chunks) * horizon

    tmp = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    if tmp.exists():
        raise FileExistsError(tmp)
    if output_path.exists():
        metadata = output_path / "brace_dataset_manifest.json"
        if metadata.is_file() and json.loads(metadata.read_text())["source_manifest_sha256"] == sha256(manifest_path):
            return json.loads(metadata.read_text())
        raise FileExistsError(f"refusing to overwrite dataset: {output_path}")

    root = zarr.group(str(tmp))
    data = root.create_group("data")
    meta = root.create_group("meta")
    try:
        for key in ("head_camera", "state", "action"):
            src = source[f"data/{key}"]
            shape = (total_frames, *src.shape[1:])
            chunks_shape = src.chunks or (min(100, total_frames), *src.shape[1:])
            dst = data.create_dataset(
                key,
                shape=shape,
                chunks=chunks_shape,
                dtype=src.dtype,
                compressor=getattr(src, "compressor", None),
                overwrite=False,
            )
            _copy_array(src, dst)
        for chunk_idx, (images, states, actions) in enumerate(chunks):
            start = expert_frames + chunk_idx * horizon
            stop = start + horizon
            data["head_camera"][start:stop] = images
            data["state"][start:stop] = states
            data["action"][start:stop] = actions

        expert_ends = np.asarray(source["meta/episode_ends"][:], dtype=np.int64)
        new_ends = expert_frames + horizon * np.arange(1, len(chunks) + 1, dtype=np.int64)
        meta.create_dataset("episode_ends", data=np.concatenate((expert_ends, new_ends)), dtype="int64")
        source_labels = np.concatenate(
            (np.zeros(expert_episodes, dtype=np.int64), np.ones(len(chunks), dtype=np.int64))
        )
        meta.create_dataset("episode_source", data=source_labels, dtype="int64")
        env_seed = np.concatenate(
            (np.full(expert_episodes, -1, dtype=np.int64), np.asarray([row["env_seed"] for row in rows], dtype=np.int64))
        )
        meta.create_dataset("episode_env_seed", data=env_seed, dtype="int64")

        payload = {
            "schema_version": 1,
            "task": rows[0]["task"],
            "dataset": rows[0]["dataset"],
            "expert_path": str(expert_path.resolve()),
            "source_manifest": str(manifest_path.resolve()),
            "source_manifest_sha256": sha256(manifest_path),
            "expert_episodes": expert_episodes,
            "chunk_episodes": len(chunks),
            "optimizer_examples_per_natural_epoch": expert_frames + len(chunks),
            "horizon": horizon,
            "n_obs_steps": n_obs_steps,
            "n_action_steps": n_action_steps,
        }
        (tmp / "brace_dataset_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, output_path)
        return payload
    except Exception:
        # Preserve failed temporary builds for diagnosis; a later run gets a new PID path.
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-zarr", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--traced-root", type=Path)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--n-obs-steps", type=int, default=3)
    parser.add_argument("--n-action-steps", type=int, default=6)
    args = parser.parse_args()
    payload = build_dataset(
        args.expert_zarr,
        args.manifest,
        args.output,
        traced_root=args.traced_root,
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
