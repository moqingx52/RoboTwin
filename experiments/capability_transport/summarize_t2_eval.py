#!/usr/bin/env python3
"""Aggregate T2 eval JSONs; derive hard-subset metrics from hard rows."""

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
    if not name.startswith("t2_eval_") or "_seed" not in name:
        raise ValueError(f"unexpected variant name: {name}")
    raw = name[len("t2_eval_") :]
    point, seed_s = raw.rsplit("_seed", 1)
    return point, int(seed_s)


def mean_sr(rows, split: str | None = None, seed_set: set[int] | None = None):
    xs = rows
    if split is not None:
        xs = [r for r in xs if r.get("split") == split]
    if seed_set is not None:
        xs = [r for r in xs if int(r["env_seed"]) in seed_set]
    xs = [r for r in xs if r.get("evaluated", True)]
    if not xs:
        return None, 0, 0
    miss = 0
    if split is not None:
        miss = sum(
            1
            for r in rows
            if r.get("split") == split
            and (seed_set is None or int(r["env_seed"]) in seed_set)
            and not r.get("evaluated", True)
        )
    return sum(bool(r["success"]) for r in xs) / len(xs), len(xs), miss


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
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--include-partial-shards",
        action="store_true",
        help="Also aggregate unfinished shard JSONs (for in-progress waves).",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = args.eval_dir / "t2_eval_summary.json"

    derived_path = args.eval_dir / "t2_eval_derived_hard_splits.json"
    panels_path = CT_DIR / "t2_eval_panels.place_container_plate.v1.json"
    if derived_path.is_file():
        derived = load_json(derived_path)
    else:
        panels = load_json(panels_path)["panels"]
        derived = {
            "within_cell_hard": panels["within_cell_hard"],
            "cross_cell_hard": panels["cross_cell_hard"],
            "right_bowl_y1_y2_hard": panels["right_bowl_y1_y2_hard"],
        }
    within = set(int(x) for x in derived["within_cell_hard"])
    cross = set(int(x) for x in derived["cross_cell_hard"])
    right = set(int(x) for x in derived["right_bowl_y1_y2_hard"])

    task_dir = args.eval_dir / "place_container_plate"
    files = sorted(p for p in task_dir.glob("t2_eval_*.json") if "_shard_" not in p.name)
    per_point: dict[str, list[dict]] = defaultdict(list)
    sources = {"merged": len(files), "partial_variants": 0}

    payloads_by_variant: dict[str, dict] = {}
    for path in files:
        payload = load_json(path)
        payloads_by_variant[payload["variant"]] = payload

    if args.include_partial_shards:
        by_var_rows: dict[str, list] = defaultdict(list)
        for path in sorted(task_dir.glob("*_shard_*_of_03.json")):
            payload = load_json(path)
            by_var_rows[payload["variant"]].extend(payload.get("rows", []))
        for variant, rows in by_var_rows.items():
            if variant in payloads_by_variant:
                continue
            payloads_by_variant[variant] = {
                "variant": variant,
                "rows": rows,
                "partial": True,
            }
            sources["partial_variants"] += 1

    for variant, payload in sorted(payloads_by_variant.items()):
        point, seed = parse_variant_name(variant)
        rows = payload.get("rows", [])
        easy_sr, easy_n, _ = mean_sr(rows, "easy")
        med_sr, med_n, _ = mean_sr(rows, "medium")
        hard_sr, hard_n, _ = mean_sr(rows, "hard")
        mem_sr, mem_n, _ = mean_sr(rows, "memorization_hard")
        within_sr, within_n, _ = mean_sr(rows, "hard", within)
        cross_sr, cross_n, _ = mean_sr(rows, "hard", cross)
        right_sr, right_n, _ = mean_sr(rows, "hard", right)
        # Fall back to explicit split labels if present (legacy shards).
        if within_n == 0:
            within_sr, within_n, _ = mean_sr(rows, "within_cell_hard")
        if cross_n == 0:
            cross_sr, cross_n, _ = mean_sr(rows, "cross_cell_hard")
        if right_n == 0:
            right_sr, right_n, _ = mean_sr(rows, "right_bowl_y1_y2_hard")

        per_point[point].append(
            {
                "seed": seed,
                "variant": variant,
                "partial": bool(payload.get("partial")),
                "easy_sr": easy_sr,
                "medium_sr": med_sr,
                "hard_sr": hard_sr,
                "within_cell_hard_sr": within_sr,
                "cross_cell_hard_sr": cross_sr,
                "right_bowl_y1_y2_hard_sr": right_sr,
                "memorization_hard_sr": mem_sr,
                "n_easy": easy_n,
                "n_medium": med_n,
                "n_hard": hard_n,
                "n_within": within_n,
                "n_cross": cross_n,
                "n_right": right_n,
                "n_mem": mem_n,
            }
        )

    point_summary = {}
    for point, rows in sorted(per_point.items()):
        rows = sorted(rows, key=lambda x: x["seed"])
        point_summary[point] = {
            "n_seeds": len(rows),
            "metrics": {
                "easy_sr_mean": mean_or_none([r["easy_sr"] for r in rows]),
                "easy_sr_std": std_or_none([r["easy_sr"] for r in rows]),
                "medium_sr_mean": mean_or_none([r["medium_sr"] for r in rows]),
                "hard_sr_mean": mean_or_none([r["hard_sr"] for r in rows]),
                "hard_sr_std": std_or_none([r["hard_sr"] for r in rows]),
                "within_cell_hard_sr_mean": mean_or_none([r["within_cell_hard_sr"] for r in rows]),
                "cross_cell_hard_sr_mean": mean_or_none([r["cross_cell_hard_sr"] for r in rows]),
                "right_bowl_y1_y2_hard_sr_mean": mean_or_none(
                    [r["right_bowl_y1_y2_hard_sr"] for r in rows]
                ),
                "memorization_hard_sr_mean": mean_or_none([r["memorization_hard_sr"] for r in rows]),
            },
            "per_seed": rows,
        }

    output = {
        "record": "capability_transport.t2_eval_summary.v2",
        "eval_dir": str(args.eval_dir),
        "sources": sources,
        "derivation": {
            "within_from_hard": True,
            "cross_from_hard": True,
            "right_bowl_from_hard": True,
        },
        "points": point_summary,
    }
    write_json(args.output, output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
