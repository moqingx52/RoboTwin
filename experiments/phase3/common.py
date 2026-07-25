#!/usr/bin/env python3
"""Shared helpers for Phase 3 CPST absorption experiments."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE1 = REPO_ROOT / "experiments" / "phase1"
PHASE3 = REPO_ROOT / "experiments" / "phase3"
DP_DIR = REPO_ROOT / "policy" / "DP"

DEFAULT_TASKS = ("place_container_plate", "dump_bin_bigbin")
DEFAULT_GPUS = tuple(range(7))

# Screen checkpoints (epoch 10 hurt B3/B4 in phase2b; include 7 as midpoint).
SCREEN_EPOCHS = (5, 7, 10)

# Main matrix A0–A4 maps to absorption variants U0–U4.
MAIN_CANDIDATES = ("A0", "A1", "A2", "A3", "A4")
FULL_EVAL_REQUIRED = ("A1", "A4")

# episode_source: 0=expert, 1=rollout_success, 2=verified_prefix
SOURCE_EXPERT = 0
SOURCE_ROLLOUT = 1
SOURCE_PREFIX = 2

# Fixed B3-style gradient budget; non-expert mass split when prefix is enabled.
DEFAULT_LAMBDA_EXPERT = 0.95
DEFAULT_LAMBDA_ROLLOUT = 0.05
DEFAULT_LAMBDA_PREFIX = 0.025
PREFIX_SPLIT_ROLLOUT = 0.025
PREFIX_SPLIT_PREFIX = 0.025

# Training-matched defaults from phase2b B3 @ epoch 5.
DEFAULT_TRAIN_EPOCHS = 5
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_BATCH_SIZE = 128
DEFAULT_CHECKPOINT_EVERY = 5

ABSORPTION_CONFIGS = {
    "U0": {
        "absorption_id": "U0",
        "dataset_variant": "expert_only",
        "diversity_mode": "none",
        "use_failure_prefix": False,
        "group_stratified": False,
        "lambda_expert": "1.0",
        "lambda_rollout": "0.0",
        "lambda_prefix": "0.0",
        "expert_ratio": "none",
        "rollout_per_batch": 0,
        "prefix_per_batch": 0,
    },
    "U1": {
        "absorption_id": "U1",
        "dataset_variant": "success_random",
        "diversity_mode": "none",
        "use_failure_prefix": False,
        "group_stratified": False,
        "lambda_expert": str(DEFAULT_LAMBDA_EXPERT),
        "lambda_rollout": str(DEFAULT_LAMBDA_ROLLOUT),
        "lambda_prefix": "0.0",
        "expert_ratio": "0.95",
        "rollout_per_batch": 6,
        "prefix_per_batch": 0,
    },
    "U2": {
        "absorption_id": "U2",
        "dataset_variant": "success_diversity",
        "diversity_mode": "seed_balanced",
        "use_failure_prefix": False,
        "group_stratified": False,
        "lambda_expert": str(DEFAULT_LAMBDA_EXPERT),
        "lambda_rollout": str(DEFAULT_LAMBDA_ROLLOUT),
        "lambda_prefix": "0.0",
        "expert_ratio": "0.95",
        "rollout_per_batch": 6,
        "prefix_per_batch": 0,
    },
    "U3": {
        "absorption_id": "U3",
        "dataset_variant": "success_plus_prefix",
        "diversity_mode": "none",
        "use_failure_prefix": True,
        "group_stratified": False,
        "lambda_expert": str(DEFAULT_LAMBDA_EXPERT),
        "lambda_rollout": str(PREFIX_SPLIT_ROLLOUT),
        "lambda_prefix": str(PREFIX_SPLIT_PREFIX),
        "expert_ratio": "0.95",
        "rollout_per_batch": 3,
        "prefix_per_batch": 3,
    },
    "U4": {
        "absorption_id": "U4",
        "dataset_variant": "success_diversity_plus_prefix",
        "diversity_mode": "seed_balanced",
        "use_failure_prefix": True,
        "group_stratified": True,
        "lambda_expert": str(DEFAULT_LAMBDA_EXPERT),
        "lambda_rollout": str(PREFIX_SPLIT_ROLLOUT),
        "lambda_prefix": str(PREFIX_SPLIT_PREFIX),
        "expert_ratio": "0.95",
        "rollout_per_batch": 3,
        "prefix_per_batch": 3,
        "group_stratified_sampling": True,
    },
}

MAIN_TO_ABSORPTION = {
    "A0": "U0",
    "A1": "U1",
    "A2": "U2",
    "A3": "U3",
    "A4": "U4",
}


def absorption_config(main_id: str) -> dict:
    absorption_id = MAIN_TO_ABSORPTION[main_id]
    cfg = dict(ABSORPTION_CONFIGS[absorption_id])
    cfg["main_id"] = main_id
    return cfg


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
    return DP_DIR / "data_phase3" / f"{task}-{suffix}.zarr"


def phase3_checkpoint_dir(task, main_id, seed, iteration=0):
    if iteration:
        return DP_DIR / "checkpoints" / f"{task}-phase3-{main_id}-{seed}-iter{iteration}"
    return DP_DIR / "checkpoints" / f"{task}-phase3-{main_id}-{seed}"


def compute_expert_steps(task, batch_size=DEFAULT_BATCH_SIZE):
    import zarr

    path = DP_DIR / "data_phase1_200" / f"{task}-expert_only.zarr"
    if not path.is_dir():
        path = DP_DIR / "data_phase3" / f"{task}-expert_only.zarr"
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


def rollout_dir(iteration=0):
    if iteration == 0:
        return PHASE1 / "rollouts_200"
    return PHASE3 / f"rollouts_iter{iteration}"


def budget_manifest_path(task, main_id, seed, iteration=0):
    return PHASE3 / "budgets" / f"{task}_{main_id}_seed{seed}_iter{iteration}.json"
