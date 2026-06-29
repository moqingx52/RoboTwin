#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from common import TASKS, write_json


VARIANTS = ("base", "expert_only", "success", "seed_balanced", "difficulty_weighted")
SPLITS = ("id_heldout", "train_seen", "hard_20")
METRICS = ("mean_sr", "solved_coverage")


def main():
    parser = argparse.ArgumentParser(description="Aggregate phase1 evaluation over training seeds.")
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--train-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = {}
    for task in args.tasks:
        summary[task] = {}
        for variant in args.variants:
            payloads = []
            for seed in args.train_seeds:
                path = args.eval_dir / f"train_seed_{seed}" / task / f"{variant}.json"
                with path.open(encoding="utf-8") as f:
                    payloads.append(json.load(f))
            variant_summary = {}
            for split in SPLITS:
                split_summary = {}
                for metric in METRICS:
                    values = [float(payload["splits"][split][metric]) for payload in payloads]
                    split_summary[metric] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                        "values": values,
                    }
                variant_summary[split] = split_summary
            summary[task][variant] = variant_summary

    write_json(args.output, summary)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
