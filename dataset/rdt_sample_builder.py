from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

from tasks.place_empty_cup.phase_labeler import label_sequence


@dataclass
class RdtSample:
    image_front: np.ndarray
    image_wrist: Optional[np.ndarray]
    proprio: np.ndarray
    instruction: str
    action_chunk: np.ndarray
    task_name: str
    episode_id: int
    success_flag: int
    phase_id: int
    phase_name: str
    lang_embed: Optional[np.ndarray] = None


def _episode_id_from_path(hdf5_path: Path) -> int:
    digits = "".join(c for c in hdf5_path.stem if c.isdigit())
    return int(digits) if digits else 0


def _decode_jpeg_rows(byte_rows: np.ndarray) -> np.ndarray:
    import cv2

    frames: list[np.ndarray] = []
    for row in byte_rows:
        encoded = bytes(row).rstrip(b"\0")
        frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("hdf5 JPEG 解码失败，请检查原始采集文件。")
        frames.append(frame[:, :, ::-1])  # BGR -> RGB
    return np.stack(frames, axis=0)


def _load_instruction(instruction_dir: Path, episode_id: int) -> str:
    path = instruction_dir / f"episode{episode_id}.json"
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    seen = payload.get("seen", [])
    if seen:
        return str(seen[0])
    unseen = payload.get("unseen", [])
    return str(unseen[0]) if unseen else ""


def _chunk_actions(actions: np.ndarray, chunk_size: int) -> list[np.ndarray]:
    chunks: list[np.ndarray] = []
    total = int(actions.shape[0])
    for start in range(0, total - chunk_size + 1):
        chunks.append(actions[start : start + chunk_size].astype(np.float32, copy=False))
    return chunks


def _read_lang_embed(embed_npz: Optional[Path], episode_id: int) -> Optional[np.ndarray]:
    if embed_npz is None or not embed_npz.exists():
        return None
    cache = np.load(embed_npz, allow_pickle=False)
    key = f"episode{episode_id}"
    if key not in cache:
        return None
    return cache[key].astype(np.float32)


def build_rdt_samples_from_episode(
    hdf5_path: Path,
    instruction_dir: Path,
    chunk_size: int,
    task_name: str,
    lang_embed_npz: Optional[Path] = None,
) -> list[RdtSample]:
    episode_id = _episode_id_from_path(hdf5_path)
    instruction = _load_instruction(instruction_dir, episode_id)

    with h5py.File(hdf5_path, "r") as fp:
        head = fp["observation"]["head_camera"]["rgb"][:]
        image_front = _decode_jpeg_rows(head)
        image_wrist = None
        if "left_wrist_camera" in fp["observation"]:
            lw = _decode_jpeg_rows(fp["observation"]["left_wrist_camera"]["rgb"][:])
            rw = _decode_jpeg_rows(fp["observation"]["right_wrist_camera"]["rgb"][:])
            image_wrist = np.stack([lw, rw], axis=1)
        proprio = fp["joint_action"]["vector"][:].astype(np.float32)

    action_chunks = _chunk_actions(proprio, chunk_size)
    phase_seq = label_sequence([a for a in proprio])
    lang_embed = _read_lang_embed(lang_embed_npz, episode_id)

    samples: list[RdtSample] = []
    for idx, chunk in enumerate(action_chunks):
        phase = phase_seq[min(idx, len(phase_seq) - 1)]
        samples.append(
            RdtSample(
                image_front=image_front[idx],
                image_wrist=image_wrist[idx] if image_wrist is not None else None,
                proprio=proprio[idx],
                instruction=instruction,
                action_chunk=chunk,
                task_name=task_name,
                episode_id=episode_id,
                success_flag=1,
                phase_id=phase.phase_id,
                phase_name=phase.phase_name,
                lang_embed=lang_embed,
            )
        )
    return samples
