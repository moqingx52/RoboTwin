#!/usr/bin/env python3
"""Aggregate per-checkpoint T2 eval JSONs into per-point summaries."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

CT_DIR = Path(__file__).resolve().parent
RUN_DIR = CT_DIR / "runs" / "20260818T065918Z_t2_train_place_container_plate"
EVAL_DIR = RUN_DIR / "eval"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def parse_variant_name(name: str) -> tuple[str, int]:
    # t2_eval_<Point>_seedNN
    if not name.startswith("t2_eval_") or "_seed" not in name:
        raise ValueError(f"unexpected variant name: {name}")
    raw = name[len("t2_eval_") :]
    point, seed_s = raw.rsplit("_seed", 1)
    return point, int(seed_s)


def metric_from_split(split_payload: dict, key: str):
    value = split_payload.get(key)
    return None if value is None else float(value)


def mean_or_none(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def std_or_none(values):
    vals = [v for v in values if v is not None]
    if len(vals) <= 1:
        return 0.0 if vals else None
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return var**0.5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--output", type=Path, default=EVAL_DIR / "t2_eval_summary.json")
    args = parser.parse_args()

    task_dir = args.eval_dir / "place_container_plate"
    files = sorted(task_dir.glob("t2_eval_*.json"))
    if not files:
        raise SystemExit(f"no eval files under {task_dir}")

    per_point: dict[str, list[dict]] = defaultdict(list)
    missing = []
    for path in files:
        payload = load_json(path)
        variant = payload["variant"]
        point, seed = parse_variant_name(variant)
        splits = payload.get("splits", {})
        row = {
            "seed": seed,
            "variant": variant,
            "file": str(path),
            "easy_sr": metric_from_split(splits.get("easy", {}), "mean_sr"),
            "medium_sr": metric_from_split(splits.get("medium", {}), "mean_sr"),
            "hard_sr": metric_from_split(splits.get("hard", {}), "mean_sr"),
            "within_cell_hard_sr": metric_from_split(splits.get("within_cell_hard", {}), "mean_sr"),
            "cross_cell_hard_sr": metric_from_split(splits.get("cross_cell_hard", {}), "mean_sr"),
            "right_bowl_y1_y2_hard_sr": metric_from_split(splits.get("right_bowl_y1_y2_hard", {}), "mean_sr"),
            "memorization_hard_sr": metric_from_split(splits.get("memorization_hard", {}), "mean_sr"),
            "hard_coverage": metric_from_split(splits.get("hard", {}), "solved_coverage"),
            "evaluated_episodes": int(
                splits.get("easy", {}).get("evaluated_episodes", 0)
                + splits.get("medium", {}).get("evaluated_episodes", 0)
                + splits.get("hard", {}).get("evaluated_episodes", 0)
            ),
            "operational_missing_episodes": int(
                splits.get("easy", {}).get("operational_missing_episodes", 0)
                + splits.get("medium", {}).get("operational_missing_episodes", 0)
                + splits.get("hard", {}).get("operational_missing_episodes", 0)
            ),
        }
        # Ensure required splits exist for summary completeness.
        required = ("easy", "medium", "hard", "within_cell_hard", "cross_cell_hard", "memorization_hard")
        if any(k not in splits for k in required):
            missing.append((variant, sorted(set(required) - set(splits.keys()))))
        per_point[point].append(row)

    if missing:
        raise SystemExit(f"incomplete eval payloads: {missing[:5]}")

    point_summary = {}
    for point, rows in sorted(per_point.items()):
        rows = sorted(rows, key=lambda x: x["seed"])
        point_summary[point] = {
            "n_seeds": len(rows),
            "metrics": {
                "easy_sr_mean": mean_or_none([r["easy_sr"] for r in rows]),
                "easy_sr_std": std_or_none([r["easy_sr"] for r in rows]),
                "medium_sr_mean": mean_or_none([r["medium_sr"] for r in rows]),
                "medium_sr_std": std_or_none([r["medium_sr"] for r in rows]),
                "hard_sr_mean": mean_or_none([r["hard_sr"] for r in rows]),
                "hard_sr_std": std_or_none([r["hard_sr"] for r in rows]),
                "within_cell_hard_sr_mean": mean_or_none([r["within_cell_hard_sr"] for r in rows]),
                "cross_cell_hard_sr_mean": mean_or_none([r["cross_cell_hard_sr"] for r in rows]),
                "right_bowl_y1_y2_hard_sr_mean": mean_or_none([r["right_bowl_y1_y2_hard_sr"] for r in rows]),
                "memorization_hard_sr_mean": mean_or_none([r["memorization_hard_sr"] for r in rows]),
                "hard_coverage_mean": mean_or_none([r["hard_coverage"] for r in rows]),
            },
            "episodes": {
                "evaluated_total": int(sum(r["evaluated_episodes"] for r in rows)),
                "operational_missing_total": int(sum(r["operational_missing_episodes"] for r in rows)),
            },
            "per_seed": rows,
        }

    output = {
        "record": "capability_transport.t2_eval_summary.v1",
        "eval_dir": str(args.eval_dir),
        "points": point_summary,
        "n_eval_files": len(files),
    }
    write_json(args.output, output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

