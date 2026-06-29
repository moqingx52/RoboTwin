#!/usr/bin/env python3
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description="Select held-out hard seeds from a DP-Base eval probe.")
    parser.add_argument("--base-eval", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = read_json(args.base_eval)
    by_seed = defaultdict(list)
    for row in payload["rows"]:
        if row["split"] == "id_heldout":
            by_seed[int(row["env_seed"])].append(bool(row["success"]))
    if len(by_seed) < args.count:
        raise RuntimeError(f"Only {len(by_seed)} ID-heldout seeds are available; need {args.count}.")

    scores = {seed: float(np.mean(values)) for seed, values in by_seed.items()}
    hard_seeds = sorted(scores, key=lambda seed: (scores[seed], seed))[: args.count]
    write_json(
        args.output,
        {
            "task_name": payload["task_name"],
            "source": str(args.base_eval),
            "hard_seeds": hard_seeds,
            "base_success_rates": {str(seed): scores[seed] for seed in hard_seeds},
        },
    )
    print(f"Wrote {args.output}: {hard_seeds}")


if __name__ == "__main__":
    main()
