#!/usr/bin/env python3
"""Verify assets required before launching screen.v1.2 GPU work."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.replay_audit import repo_path


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_hdf5(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return len(list(directory.glob("*.hdf5")))


def check_assets(*, task: str, run_label: str, traced_rollout_dir: Path) -> dict:
    base_ckpt = REPO_ROOT / "policy/DP/checkpoints" / f"{task}-demo_clean-200-0/600.ckpt"
    expert = REPO_ROOT / "policy/DP/data_phase1_200" / f"{task}-expert_only.zarr"
    b1_manifest = REPO_ROOT / "experiments/brace/datasets" / f"{run_label}_B1.jsonl"
    n1_manifest = REPO_ROOT / "experiments/brace/datasets" / f"{run_label}_N1.jsonl"
    hard_seeds = REPO_ROOT / "experiments/phase1/eval_results_200/hard_eval_seeds" / f"{task}.json"
    traced = traced_rollout_dir / task
    success_dir = traced / "successes"
    failure_dir = traced / "failures"
    success_count = count_hdf5(success_dir)
    failure_count = count_hdf5(failure_dir)
    checks = {
        "base_checkpoint": {"path": str(base_ckpt), "exists": base_ckpt.is_file(), "sha256": sha256_file(base_ckpt)},
        "expert_zarr": {"path": str(expert), "exists": expert.is_dir()},
        "b1_manifest": {"path": str(b1_manifest), "exists": b1_manifest.is_file()},
        "n1_manifest": {"path": str(n1_manifest), "exists": n1_manifest.is_file()},
        "hard_seeds": {"path": str(hard_seeds), "exists": hard_seeds.is_file()},
        "traced_rollouts": {
            "path": str(traced),
            "exists": traced.is_dir(),
            "success_count": success_count,
            "failure_count": failure_count,
            "ready": traced.is_dir() and success_count > 0 and failure_count > 0,
        },
    }
    checks["ready"] = all(
        [
            checks["base_checkpoint"]["exists"],
            checks["expert_zarr"]["exists"],
            checks["b1_manifest"]["exists"],
            checks["n1_manifest"]["exists"],
            checks["hard_seeds"]["exists"],
            checks["traced_rollouts"]["ready"],
        ]
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="place_container_plate")
    parser.add_argument("--run-label", default="place_pilot_v2.3")
    parser.add_argument("--traced-rollout-dir", type=Path, default=Path("experiments/brace/rollouts_traced"))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    payload = check_assets(
        task=args.task,
        run_label=args.run_label,
        traced_rollout_dir=repo_path(args.traced_rollout_dir),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.prepare_only and payload["ready"]:
        print("prepare-only dry run: assets look sufficient for screen.v1.2")
    return 0 if payload["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
