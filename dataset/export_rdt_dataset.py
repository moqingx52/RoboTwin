from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dataset.rdt_sample_builder import build_rdt_samples_from_episode


def _iter_episodes(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("episode*.hdf5"))


def _write_manifest_row(out_fp, payload: dict) -> None:
    out_fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def export_dataset(
    source_root: Path,
    output_root: Path,
    task_name: str,
    chunk_size: int,
    lang_embed_npz: Path | None,
) -> None:
    data_dir = source_root / "data"
    instruction_dir = source_root / "instructions"
    episodes = _iter_episodes(data_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"

    with manifest_path.open("w", encoding="utf-8") as manifest_fp:
        for hdf5_path in episodes:
            samples = build_rdt_samples_from_episode(
                hdf5_path=hdf5_path,
                instruction_dir=instruction_dir,
                chunk_size=chunk_size,
                task_name=task_name,
                lang_embed_npz=lang_embed_npz,
            )
            for i, sample in enumerate(samples):
                sample_file = (
                    output_root
                    / f"episode{sample.episode_id}_step{i:04d}.npz"
                )
                np.savez_compressed(
                    sample_file,
                    image_front=sample.image_front,
                    image_wrist=sample.image_wrist
                    if sample.image_wrist is not None
                    else np.zeros((0,), dtype=np.uint8),
                    proprio=sample.proprio,
                    instruction=sample.instruction,
                    action_chunk=sample.action_chunk,
                    task_name=sample.task_name,
                    episode_id=np.int32(sample.episode_id),
                    success_flag=np.int32(sample.success_flag),
                    phase_id=np.int32(sample.phase_id),
                    phase_name=sample.phase_name,
                    lang_embed=sample.lang_embed
                    if sample.lang_embed is not None
                    else np.zeros((0,), dtype=np.float32),
                )
                _write_manifest_row(
                    manifest_fp,
                    {
                        "file": sample_file.name,
                        "task_name": sample.task_name,
                        "episode_id": sample.episode_id,
                        "phase_id": sample.phase_id,
                        "phase_name": sample.phase_name,
                        "success_flag": sample.success_flag,
                    },
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 RDT 第一阶段训练样本。")
    parser.add_argument("--source-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--task-name", type=str, default="place_empty_cup")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--lang-embed-npz", type=str, default="")
    args = parser.parse_args()

    lang_embed_npz = Path(args.lang_embed_npz) if args.lang_embed_npz else None
    export_dataset(
        source_root=Path(args.source_root),
        output_root=Path(args.output_root),
        task_name=args.task_name,
        chunk_size=int(args.chunk_size),
        lang_embed_npz=lang_embed_npz,
    )


if __name__ == "__main__":
    main()
