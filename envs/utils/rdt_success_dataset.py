import json
import os
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np


CAMERA_KEYS = {
    "cam_high": "cam_high",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def _next_episode_index(dataset_dir: Path) -> int:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    max_idx = -1
    for path in dataset_dir.glob("episode_*"):
        if not path.is_dir():
            continue
        try:
            max_idx = max(max_idx, int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max_idx + 1


def _encode_image(image: np.ndarray) -> bytes:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {arr.shape}")
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode image as jpeg")
    return encoded.tobytes()


def _write_encoded_images(group: h5py.Group, name: str, images: list[np.ndarray]) -> None:
    encoded = [_encode_image(image) for image in images]
    max_len = max(len(buf) for buf in encoded)
    dtype = f"S{max_len}"
    data = np.asarray(encoded, dtype=dtype)
    group.create_dataset(name, data=data)


def write_rdt_success_episode(
    dataset_dir: str | os.PathLike[str],
    trace: list[dict[str, Any]],
    *,
    instruction: str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write one successful RDT episode in the processed HDF5 layout."""

    if not trace:
        raise ValueError("Cannot write an empty RDT success trace")

    root = Path(dataset_dir)
    episode_idx = _next_episode_index(root)
    episode_dir = root / f"episode_{episode_idx}"
    episode_dir.mkdir(parents=True, exist_ok=False)
    hdf5_path = episode_dir / f"episode_{episode_idx}.hdf5"

    actions = np.stack([np.asarray(step["action"], dtype=np.float32) for step in trace])
    qpos = np.stack([np.asarray(step["qpos"], dtype=np.float32) for step in trace])
    left_dims = np.full((len(trace),), 6, dtype=np.int64)
    right_dims = np.full((len(trace),), 6, dtype=np.int64)

    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("action", data=actions)
        obs_group = f.create_group("observations")
        obs_group.create_dataset("qpos", data=qpos)
        obs_group.create_dataset("left_arm_dim", data=left_dims)
        obs_group.create_dataset("right_arm_dim", data=right_dims)
        img_group = obs_group.create_group("images")
        _write_encoded_images(img_group, "cam_high", [step["cam_high"] for step in trace])
        _write_encoded_images(
            img_group,
            "cam_left_wrist",
            [step["cam_left_wrist"] for step in trace],
        )
        _write_encoded_images(
            img_group,
            "cam_right_wrist",
            [step["cam_right_wrist"] for step in trace],
        )

    instruction_payload = {"seen": [instruction], "instruction": instruction}
    with open(episode_dir / "instruction.json", "w", encoding="utf-8") as f:
        json.dump(instruction_payload, f, indent=2, ensure_ascii=False)

    meta = dict(metadata or {})
    meta.update(
        {
            "episode_index": episode_idx,
            "instruction": instruction,
            "num_steps": len(trace),
            "hdf5_path": str(hdf5_path),
        }
    )
    with open(episode_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=_json_default, ensure_ascii=False)

    return hdf5_path
