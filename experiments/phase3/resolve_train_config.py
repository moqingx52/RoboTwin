#!/usr/bin/env python3
"""Resolve the canonical matched-budget settings for one Phase 3 variant."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import absorption_config, compute_expert_steps  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--main-id", required=True, choices=("A0", "A1", "A2", "A3", "A4"))
    args = parser.parse_args()

    cfg = absorption_config(args.main_id)
    values = (
        compute_expert_steps(args.task),
        cfg["dataset_variant"],
        "source_separated",
        cfg["lambda_expert"],
        cfg["lambda_rollout"],
        cfg["lambda_prefix"],
        cfg["rollout_per_batch"],
        cfg["prefix_per_batch"],
        str(bool(cfg.get("group_stratified_sampling", False))).lower(),
    )
    for value in values:
        print(value)


if __name__ == "__main__":
    main()
