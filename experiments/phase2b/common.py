#!/usr/bin/env python3
"""Shared helpers for Phase 2b."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE1 = REPO_ROOT / "experiments" / "phase1"
PHASE2B = REPO_ROOT / "experiments" / "phase2b"
DP_DIR = REPO_ROOT / "policy" / "DP"

DEFAULT_TASKS = ("place_container_plate", "dump_bin_bigbin")
DEFAULT_GPUS = tuple(range(7))

DIAGNOSE_VARIANTS = ("A0_base", "A1_expert_norm", "A2_mixed_norm")
SCREEN_CANDIDATES = ("B0", "B1", "B2", "B3", "B4")
SCREEN_EPOCHS = (5, 10, 20)

CANDIDATE_CONFIGS = {
    "B0": {
        "dataset_suffix": "expert_only",
        "expert_ratio": "none",
        "loss_mode": "pooled",
        "lambda_expert": "1.0",
        "lambda_rollout": "0.0",
    },
    "B1": {
        "dataset_suffix": "success",
        "expert_ratio": "0.9",
        "loss_mode": "pooled",
        "lambda_expert": "1.0",
        "lambda_rollout": "0.0",
    },
    "B2": {
        "dataset_suffix": "success",
        "expert_ratio": "0.9",
        "loss_mode": "source_separated",
        "lambda_expert": "0.9",
        "lambda_rollout": "0.1",
    },
    "B3": {
        "dataset_suffix": "success",
        "expert_ratio": "0.95",
        "loss_mode": "source_separated",
        "lambda_expert": "0.95",
        "lambda_rollout": "0.05",
    },
    "B4": {
        "dataset_suffix": "success",
        "expert_ratio": "0.9",
        "loss_mode": "source_separated",
        "lambda_expert": "0.95",
        "lambda_rollout": "0.05",
    },
}


def now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def base_checkpoint(task):
    return DP_DIR / "checkpoints" / f"{task}-demo_clean-200-0" / "600.ckpt"


def dataset_path(task, suffix):
    return DP_DIR / "data_phase1_200" / f"{task}-{suffix}.zarr"


def phase2b_checkpoint_dir(task, candidate, seed):
    return DP_DIR / "checkpoints" / f"{task}-phase2b-{candidate}-{seed}"


def diagnose_ckpt_path(task, variant):
    if variant == "A0_base":
        return base_checkpoint(task)
    return PHASE2B / "diagnose" / task / f"{variant}.ckpt"


def compute_expert_steps(task, batch_size=128):
    import zarr

    path = dataset_path(task, "expert_only")
    root = zarr.open(str(path), mode="r")
    ends = root["meta/episode_ends"][:]
    if len(ends) < 2:
        raise RuntimeError(f"Not enough episodes in {path}")
    steps = int(ends[-2]) // batch_size
    if steps <= 0:
        raise RuntimeError(f"Not enough samples in {path}")
    return steps


def base_eval_path(task, train_seed=0):
    return PHASE1 / "eval_results_200" / f"train_seed_{train_seed}" / task / "base.json"


def hard_seeds_file(task):
    return PHASE1 / "eval_results_200" / "hard_eval_seeds" / f"{task}.json"


def repo_path(*parts):
    return REPO_ROOT.joinpath(*parts)
