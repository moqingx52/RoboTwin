#!/usr/bin/env python3
"""Reproducible forensic analysis for BRACE developmental screens."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.replay_audit import git_commit, read_json, repo_path, write_json_atomic


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def episode_outcomes(eval_path: Path, split: str) -> dict[int, bool]:
    payload = read_json(eval_path)
    rows = payload.get("rows", [])
    return {
        int(row["env_seed"]): bool(row.get("success", row.get("task_success", False)))
        for row in rows
        if row.get("split") == split
    }


def paired_flips(left: dict[int, bool], right: dict[int, bool]) -> dict[str, Any]:
    seeds = sorted(set(left) & set(right))
    left_only = sum(1 for seed in seeds if left[seed] and not right[seed])
    right_only = sum(1 for seed in seeds if right[seed] and not left[seed])
    both = sum(1 for seed in seeds if left[seed] and right[seed])
    neither = sum(1 for seed in seeds if not left[seed] and not right[seed])
    return {
        "paired_seeds": len(seeds),
        "left_only": left_only,
        "right_only": right_only,
        "both_success": both,
        "neither_success": neither,
        "left_sr": (left_only + both) / len(seeds) if seeds else 0.0,
        "right_sr": (right_only + both) / len(seeds) if seeds else 0.0,
    }


def mcnemar_exact(left_only: int, right_only: int) -> float:
    n = left_only + right_only
    if n == 0:
        return 1.0
    # two-sided exact binomial
    k = min(left_only, right_only)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / (2**n))


def paired_bootstrap(left: dict[int, bool], right: dict[int, bool], *, samples: int = 100_000, seed: int = 0) -> dict[str, float]:
    seeds = sorted(set(left) & set(right))
    if not seeds:
        return {"low": 0.0, "high": 0.0, "mean": 0.0}
    rng = random.Random(seed)
    diffs = []
    for _ in range(samples):
        draw = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        left_sr = sum(left[s] for s in draw) / len(draw)
        right_sr = sum(right[s] for s in draw) / len(draw)
        diffs.append(left_sr - right_sr)
    diffs.sort()
    return {
        "mean": float(sum(diffs) / len(diffs)),
        "low": float(diffs[int(0.025 * len(diffs))]),
        "high": float(diffs[int(0.975 * len(diffs)) - 1]),
    }


def manifest_audit(b1_path: Path, n1_path: Path) -> dict[str, Any]:
    b1 = read_jsonl(b1_path)
    n1 = read_jsonl(n1_path)
    b1_seeds = sorted({int(row["env_seed"]) for row in b1})
    n1_seeds = sorted({int(row["env_seed"]) for row in n1})
    return {
        "b1_chunks": len(b1),
        "n1_chunks": len(n1),
        "b1_unique_env_seeds": len(b1_seeds),
        "n1_unique_env_seeds": len(n1_seeds),
        "shared_env_seeds": sorted(set(b1_seeds) & set(n1_seeds)),
        "b1_only_env_seeds": sorted(set(b1_seeds) - set(n1_seeds)),
        "n1_only_env_seeds": sorted(set(n1_seeds) - set(b1_seeds)),
        "b1_chunk_index_hist": dict(Counter(int(row["branch_chunk_index"]) for row in b1)),
        "n1_chunk_index_hist": dict(Counter(int(row["branch_chunk_index"]) for row in n1)),
        "matched_pair_ids": sorted({row.get("matched_pair_id") for row in n1 if row.get("matched_pair_id")}),
    }


def parse_anchor_logs(log_paths: list[Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in log_paths:
        if not path.is_file():
            continue
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if not rows:
            continue
        last = rows[-1]
        key = path.parent.name
        groups = {}
        for field, value in last.items():
            if field.startswith("brace_constraint/"):
                groups.setdefault(field.split("/", 1)[1], {})["constraint_end"] = value
            if field.startswith("brace_dual/"):
                groups.setdefault(field.split("/", 1)[1], {})["dual_end"] = value
        result[key] = {"groups": groups, "steps": len(rows)}
    return result


def build_report(
    *,
    run_dir: Path,
    task: str,
    comparisons: list[tuple[str, Path, str, Path]],
    b1_manifest: Path | None,
    n1_manifest: Path | None,
    anchor_log_glob: str = "**/logs.json.txt",
) -> dict[str, Any]:
    eval_dir = run_dir / "eval" / task
    paired = {}
    for left_name, left_eval, right_name, right_eval in comparisons:
        for split in ("id_heldout", "train_seen", "hard_20"):
            left = episode_outcomes(eval_dir / left_eval, split)
            right = episode_outcomes(eval_dir / right_eval, split)
            flips = paired_flips(left, right)
            flips["mcnemar_p"] = mcnemar_exact(flips["left_only"], flips["right_only"])
            flips["bootstrap"] = paired_bootstrap(left, right)
            paired[f"{left_name}_vs_{right_name}:{split}"] = flips

    manifest = None
    if b1_manifest and n1_manifest and b1_manifest.is_file() and n1_manifest.is_file():
        manifest = manifest_audit(b1_manifest, n1_manifest)

    summary_path = run_dir / "summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else None
    anchor_logs = sorted((REPO_ROOT / "policy/DP/data/outputs").glob(anchor_log_glob))
    return {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "task": task,
        "git_commit": git_commit(),
        "summary": summary,
        "paired_comparisons": paired,
        "manifest_audit": manifest,
        "anchor_training_epochs": parse_anchor_logs(anchor_logs[-18:]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task", default="place_container_plate")
    parser.add_argument("--b1-manifest", type=Path)
    parser.add_argument("--n1-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = repo_path(args.run_dir)
    comparisons = [
        ("B1", "B1_epoch10.json", "N1", "N1_epoch5.json"),
        ("B2", "B2_epoch3.json", "N1", "N1_epoch5.json"),
        ("B3", "B3_epoch3.json", "B1", "B1_epoch10.json"),
    ]
    report = build_report(
        run_dir=run_dir,
        task=args.task,
        comparisons=comparisons,
        b1_manifest=repo_path(args.b1_manifest) if args.b1_manifest else None,
        n1_manifest=repo_path(args.n1_manifest) if args.n1_manifest else None,
    )
    output = repo_path(args.output) if args.output else run_dir / "forensic_report.json"
    write_json_atomic(output, report)
    print(json.dumps({"output": str(output), "paired_keys": list(report["paired_comparisons"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
