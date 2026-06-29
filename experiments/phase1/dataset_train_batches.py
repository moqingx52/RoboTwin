#!/usr/bin/env python3
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Print natural DP training batches for a phase1 dataset.")
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    import zarr

    root = zarr.open(str(args.zarr_path), mode="r")
    episode_ends = root["meta/episode_ends"][:]
    if len(episode_ends) < 2:
        raise RuntimeError("Dataset needs at least two episodes because one is reserved for validation.")
    # RobotImageDataset currently reserves the final episode for validation.
    # With the phase1 horizon/padding, each trajectory frame contributes one
    # sequence sample, so the final training endpoint is the sample count.
    train_samples = int(episode_ends[-2])
    batches = train_samples // args.batch_size
    if batches <= 0:
        raise RuntimeError(f"Only {train_samples} training samples for batch size {args.batch_size}.")
    print(batches)


if __name__ == "__main__":
    main()
