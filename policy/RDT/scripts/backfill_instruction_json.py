#!/usr/bin/env python3
"""Backfill episode instruction.json for processed RDT datasets (no T5 re-encode)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from process_data import write_episode_instruction_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy RoboTwin data/.../instructions/episode*.json into processed episode dirs."
    )
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        required=True,
        help="e.g. processed_data/place_empty_cup-demo_clean-20 or training_data/<run>/place_empty_cup-demo_clean-20",
    )
    parser.add_argument("--episode-num", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rdt_root = Path(__file__).resolve().parents[1]
    instructions_root = (
        (rdt_root / "../../data" / args.task_name / args.task_config / "instructions")
        .resolve()
    )
    processed_dir = args.processed_dir
    if not processed_dir.is_absolute():
        processed_dir = Path.cwd() / processed_dir

    for idx in range(args.episode_num):
        src = instructions_root / f"episode{idx}.json"
        target_dir = processed_dir / f"episode_{idx}"
        out = write_episode_instruction_json(str(src), str(target_dir))
        print(f"episode_{idx}: wrote {out}")


if __name__ == "__main__":
    main()
