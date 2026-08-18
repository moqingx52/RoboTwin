#!/usr/bin/env python3
"""Build T2 mixed zarrs: frozen Base200 rehearsal + new-source hdf5 trajectories.

Zero uses the original Base200 zarr (no copy). Mixed arms concatenate Base200
with Q new episodes and write meta/episode_source (0=Base200, 1=new source).

Usage (inside the cloud container, from /workspace/RoboTwin):
    python experiments/capability_transport/build_t2_mixtures.py
    python experiments/capability_transport/build_t2_mixtures.py --point Expert-Cover-12
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import zarr

CT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CT_DIR))

from common import episode_to_arrays, repo_path  # noqa: E402
from t2_train_lib import (  # noqa: E402
    DEFAULT_MIXTURE_DIR,
    load_launch_config,
    missing_new_source_paths,
    mixture_zarr_path,
    resolve_new_source_episodes,
    sha256_file,
    write_json,
)

COPY_FRAMES = 256
SOURCE_BASE200 = 0
SOURCE_NEW = 1


def _compressor(arr):
    return getattr(arr, "compressor", None)


def _copy_concat_array(src_arr, extra, dest_group, name: str) -> None:
    n0 = int(src_arr.shape[0])
    n1 = 0 if extra is None else int(extra.shape[0])
    if extra is not None and tuple(extra.shape[1:]) != tuple(src_arr.shape[1:]):
        raise ValueError(
            f"{name} new shape {extra.shape} does not match Base200 {src_arr.shape}"
        )
    out = dest_group.create_dataset(
        name,
        shape=(n0 + n1,) + tuple(src_arr.shape[1:]),
        chunks=src_arr.chunks,
        dtype=src_arr.dtype,
        compressor=_compressor(src_arr),
        overwrite=True,
    )
    for start in range(0, n0, COPY_FRAMES):
        end = min(n0, start + COPY_FRAMES)
        out[start:end] = src_arr[start:end]
    if extra is not None and n1:
        out[n0:] = extra


def build_mixed_zarr(
    point: str,
    mixture_cfg: dict,
    base_zarr: Path,
    out_zarr: Path,
    force: bool = False,
) -> dict:
    episodes = resolve_new_source_episodes(point, mixture_cfg)
    expected_q = int(mixture_cfg["Q"])
    if len(episodes) != expected_q:
        raise SystemExit(f"{point}: expected Q={expected_q} hdf5 files, got {len(episodes)}")
    missing = [str(rec["path"]) for rec in episodes if not rec["path"].is_file()]
    if missing:
        raise SystemExit(f"{point}: missing hdf5:\n" + "\n".join(missing))

    manifest_path = out_zarr / "t2_mixture_manifest.json"
    hdf5_sha = {rec["relpath"]: sha256_file(rec["path"]) for rec in episodes}
    if out_zarr.exists() and manifest_path.is_file() and not force:
        existing = json_read(manifest_path)
        if existing.get("new_source_hdf5_sha256") == hdf5_sha and existing.get("Q") == expected_q:
            print(f"skip existing mixture {out_zarr}")
            return existing

    if out_zarr.exists():
        shutil.rmtree(out_zarr)

    new_head, new_state, new_action = [], [], []
    new_lengths = []
    new_seeds = []
    for rec in episodes:
        head, state, action = episode_to_arrays(rec["path"])
        if head.shape[0] != state.shape[0] or state.shape[0] != action.shape[0]:
            raise SystemExit(
                f"{rec['path']}: mismatched lengths "
                f"head={head.shape[0]} state={state.shape[0]} action={action.shape[0]}"
            )
        new_head.append(head)
        new_state.append(state)
        new_action.append(action)
        new_lengths.append(int(state.shape[0]))
        new_seeds.append(int(rec["env_seed"]))
        print(f"  {point}: {rec['relpath']} frames={state.shape[0]}")

    extra_head = np.concatenate(new_head, axis=0)
    extra_state = np.concatenate(new_state, axis=0)
    extra_action = np.concatenate(new_action, axis=0)

    src = zarr.open(str(base_zarr), mode="r")
    src_data = src["data"]
    src_meta = src["meta"]
    base_ends = np.asarray(src_meta["episode_ends"][:], dtype=np.int64)
    n_base_eps = int(base_ends.shape[0])
    if n_base_eps != 200:
        raise SystemExit(f"Base200 zarr has {n_base_eps} episodes, expected 200")

    out_zarr.mkdir(parents=True, exist_ok=True)
    root = zarr.group(str(out_zarr))
    data = root.create_group("data")
    meta = root.create_group("meta")

    _copy_concat_array(src_data["head_camera"], extra_head, data, "head_camera")
    _copy_concat_array(src_data["state"], extra_state, data, "state")
    _copy_concat_array(src_data["action"], extra_action, data, "action")

    offset = int(base_ends[-1])
    new_ends = offset + np.cumsum(np.asarray(new_lengths, dtype=np.int64))
    episode_ends = np.concatenate([base_ends, new_ends]).astype(np.int64)
    episode_source = np.concatenate(
        [
            np.full(n_base_eps, SOURCE_BASE200, dtype=np.int64),
            np.full(expected_q, SOURCE_NEW, dtype=np.int64),
        ]
    )
    episode_env_seed = np.concatenate(
        [
            np.full(n_base_eps, -1, dtype=np.int64),
            np.asarray(new_seeds, dtype=np.int64),
        ]
    )
    compressor = _compressor(src_meta["episode_ends"])
    meta.create_dataset(
        "episode_ends", data=episode_ends, dtype="int64", compressor=compressor, overwrite=True
    )
    meta.create_dataset(
        "episode_source", data=episode_source, dtype="int64", compressor=compressor, overwrite=True
    )
    meta.create_dataset(
        "episode_env_seed",
        data=episode_env_seed,
        dtype="int64",
        compressor=compressor,
        overwrite=True,
    )

    manifest = {
        "record": "capability_transport.t2_mixture.v1",
        "point": point,
        "Q": expected_q,
        "rho": mixture_cfg["rho"],
        "w_traj": mixture_cfg.get("w_traj"),
        "base_zarr": str(base_zarr),
        "n_base_episodes": n_base_eps,
        "n_new_episodes": expected_q,
        "n_base_frames": offset,
        "n_new_frames": int(sum(new_lengths)),
        "episode_source": "0=Base200 rehearsal, 1=new T2 source",
        "new_source_run_dir": mixture_cfg["source_run_dir"],
        "new_source_hdf5": [rec["relpath"] for rec in episodes],
        "new_source_env_seeds": new_seeds,
        "new_source_hdf5_sha256": hdf5_sha,
        "note": mixture_cfg.get("note"),
    }
    write_json(manifest_path, manifest)
    print(f"wrote {out_zarr}  episodes={n_base_eps + expected_q}  frames={int(episode_ends[-1])}")
    return manifest


def json_read(path: Path) -> dict:
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T2 Base200+new-source zarr mixtures")
    parser.add_argument("--point", action="append", dest="points", default=None)
    parser.add_argument("--mixture-dir", type=Path, default=DEFAULT_MIXTURE_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg, launch_sha = load_launch_config()
    base_zarr = repo_path(cfg["base_policy"]["zarr"])
    if not base_zarr.is_dir():
        raise SystemExit(f"missing Base200 zarr: {base_zarr}")
    ckpt = repo_path(cfg["base_policy"]["ckpt"])
    ckpt_sha = sha256_file(ckpt)
    if ckpt_sha != cfg["base_policy"]["ckpt_sha256"]:
        raise SystemExit(
            f"Base200 ckpt sha256 {ckpt_sha} != launch_config {cfg['base_policy']['ckpt_sha256']}"
        )

    points = args.points or list(cfg["training_grid"]["points"])
    args.mixture_dir.mkdir(parents=True, exist_ok=True)
    print(f"launch_config sha256 {launch_sha}")
    for point in points:
        mixture_cfg = cfg["mixtures"][point]
        missing = missing_new_source_paths(point, mixture_cfg)
        if missing:
            raise SystemExit(f"{point}: missing hdf5:\n" + "\n".join(missing))
        out = mixture_zarr_path(point, args.mixture_dir)
        if point == "Zero":
            print(f"Zero: reuse {out} (no episode_source; expert_ratio=null)")
            continue
        print(f"building {point} -> {out}")
        build_mixed_zarr(point, mixture_cfg, base_zarr, out, force=args.force)


if __name__ == "__main__":
    main()
