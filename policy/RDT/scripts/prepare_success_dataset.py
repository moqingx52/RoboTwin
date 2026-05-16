import argparse
import json
from pathlib import Path

from encode_lang_batch_once import encode_lang


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate per-episode RDT language embeddings for collected success HDF5 data."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("training_data/place_empty_cup_rdt_success"),
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--desc-type", default="seen")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    episode_dirs = sorted(
        path for path in dataset_dir.glob("episode_*") if path.is_dir()
    )
    if not episode_dirs:
        raise RuntimeError(f"No episode_* directories found in {dataset_dir}")

    prepared = 0
    tokenizer = None
    text_encoder = None
    for episode_dir in episode_dirs:
        hdf5_files = list(episode_dir.glob("*.hdf5"))
        if not hdf5_files:
            raise RuntimeError(f"No hdf5 file found in {episode_dir}")
        instruction_json = episode_dir / "instruction.json"
        if not instruction_json.is_file():
            metadata_json = episode_dir / "metadata.json"
            if not metadata_json.is_file():
                raise RuntimeError(f"No instruction.json or metadata.json in {episode_dir}")
            with open(metadata_json, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            instruction = metadata.get(
                "instruction",
                "Place the empty cup to the target area.",
            )
            with open(instruction_json, "w", encoding="utf-8") as f:
                json.dump({"seen": [instruction], "instruction": instruction}, f, indent=2)

        tokenizer, text_encoder = encode_lang(
            DATA_FILE_PATH=str(instruction_json),
            TARGET_DIR=str(episode_dir),
            GPU=str(args.gpu),
            desc_type=str(args.desc_type),
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )
        prepared += 1

    print(
        f"[prepare-success-dataset] prepared={prepared} dataset_dir={dataset_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
