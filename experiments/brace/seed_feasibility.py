#!/usr/bin/env python3
"""Expert-script seed feasibility scanning and expert_demo cohort selection."""

from __future__ import annotations

import importlib
import json
import traceback
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"

EXPERT_DEMO_SELECTION_RULE = "first_n_solvable_in_manifest_order"
FEASIBILITY_CRITERION = "expert_script_simulator_solvability_only"
DEFAULT_TASK_CONFIG = "demo_clean"
DEFAULT_CANDIDATE_PARTITION = "rollout_train"
DEFAULT_EXPERT_DEMO_COUNT = 50

TASK_STATUS_PASSED = "passed"
TASK_STATUS_INSUFFICIENT = "insufficient_solvable"
TASK_STATUS_EXPERT_EXCEPTION = "expert_exception"
TASK_STATUS_PENDING = "pending"

SEED_NOT_SOLVABLE_ERROR_TYPES = {
    "pre_motion_validation_failed",
    "simulator_unstable",
}

EXPERT_EXCEPTION_ERROR_TYPES = {
    "expert_assertion_failed",
    "missing_asset",
    "expert_implementation_error",
    "expert_probe_error",
}


def classify_probe_error(exc: BaseException) -> tuple[str, str]:
    message = str(exc).strip() or exc.__class__.__name__
    name = exc.__class__.__name__
    if name == "RuntimeError" and "pre-motion validation" in message:
        return "pre_motion_validation_failed", message
    if name == "UnStableError":
        return "simulator_unstable", message
    if name == "AssertionError":
        return "expert_assertion_failed", message
    if name in {"ImportError", "ModuleNotFoundError", "AttributeError", "TypeError"}:
        return "expert_implementation_error", message
    if name == "FileNotFoundError":
        return "missing_asset", message
    return "expert_probe_error", message


def is_expert_exception_error(error_type: str | None) -> bool:
    return error_type in EXPERT_EXCEPTION_ERROR_TYPES


def derive_task_status(
    probe_results: list[dict[str, Any]],
    *,
    candidate_count: int,
    required_count: int = DEFAULT_EXPERT_DEMO_COUNT,
) -> str:
    if len(probe_results) < candidate_count:
        return TASK_STATUS_PENDING
    solvable_count = sum(1 for row in probe_results if row.get("passed") is True)
    if solvable_count >= required_count:
        return TASK_STATUS_PASSED
    expert_exception_count = sum(
        1 for row in probe_results if is_expert_exception_error(row.get("error_type"))
    )
    if expert_exception_count > 0:
        return TASK_STATUS_EXPERT_EXCEPTION
    return TASK_STATUS_INSUFFICIENT


def select_expert_demo_seeds(
    candidate_seeds: list[int],
    probe_results: list[dict[str, Any]],
    *,
    required_count: int = DEFAULT_EXPERT_DEMO_COUNT,
    rule: str = EXPERT_DEMO_SELECTION_RULE,
) -> list[int]:
    if rule != EXPERT_DEMO_SELECTION_RULE:
        raise ValueError(f"unsupported selection rule: {rule}")
    by_seed = {int(row["seed"]): row for row in probe_results}
    selected: list[int] = []
    for seed in candidate_seeds:
        row = by_seed.get(int(seed))
        if row and row.get("passed") is True:
            selected.append(int(seed))
        if len(selected) >= required_count:
            break
    return selected


def build_feasibility_evidence(
    *,
    task: str,
    candidate_seeds: list[int],
    probe_results: list[dict[str, Any]],
    task_config: str = DEFAULT_TASK_CONFIG,
    candidate_partition: str = DEFAULT_CANDIDATE_PARTITION,
    required_count: int = DEFAULT_EXPERT_DEMO_COUNT,
    rule: str = EXPERT_DEMO_SELECTION_RULE,
) -> dict[str, Any]:
    selected = select_expert_demo_seeds(
        candidate_seeds,
        probe_results,
        required_count=required_count,
        rule=rule,
    )
    solvable_count = sum(1 for row in probe_results if row.get("passed") is True)
    expert_exception_count = sum(
        1 for row in probe_results if is_expert_exception_error(row.get("error_type"))
    )
    seed_not_solvable_count = sum(
        1
        for row in probe_results
        if not row.get("passed") and row.get("error_type") in SEED_NOT_SOLVABLE_ERROR_TYPES
    )
    task_status = derive_task_status(
        probe_results,
        candidate_count=len(candidate_seeds),
        required_count=required_count,
    )
    return {
        "schema_version": 1,
        "stage": "seed_feasibility_scan",
        "task": task,
        "criterion": FEASIBILITY_CRITERION,
        "learning_policy_performance_consulted": False,
        "task_config": task_config,
        "candidate_partition": candidate_partition,
        "candidate_count": len(candidate_seeds),
        "selection_rule": rule,
        "expert_demo_count": required_count,
        "expert_demo_seeds": selected,
        "solvable_count_in_candidates": solvable_count,
        "expert_exception_count": expert_exception_count,
        "seed_not_solvable_count": seed_not_solvable_count,
        "task_status": task_status,
        "passed": task_status == TASK_STATUS_PASSED,
        "results": probe_results,
    }


def validate_feasibility_evidence(evidence: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if evidence.get("criterion") != FEASIBILITY_CRITERION:
        errors.append("unexpected feasibility criterion")
    if evidence.get("learning_policy_performance_consulted") is not False:
        errors.append("learning_policy_performance_consulted must be false")
    if evidence.get("selection_rule") != EXPERT_DEMO_SELECTION_RULE:
        errors.append("unexpected expert_demo selection rule")
    required = evidence.get("expert_demo_count", DEFAULT_EXPERT_DEMO_COUNT)
    selected = evidence.get("expert_demo_seeds", [])
    if not isinstance(selected, list):
        errors.append("expert_demo_seeds must be a list")
    elif evidence.get("passed") is True and len(selected) != required:
        errors.append("passed evidence must contain exactly expert_demo_count seeds")
    results = evidence.get("results", [])
    if not isinstance(results, list) or not results:
        errors.append("results must be a non-empty list")
    return errors


def apply_provisional_expert_demo_to_manifest(manifest: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    """Record provisional expert_demo selection; manifest stays candidate_unvalidated."""
    if evidence.get("task_status") != TASK_STATUS_PASSED:
        raise ValueError("provisional manifest update requires task_status=passed")
    errors = validate_feasibility_evidence(evidence)
    if errors:
        raise ValueError("; ".join(errors))
    rollout_train = manifest.get("partitions", {}).get("rollout_train", [])
    selected = [int(seed) for seed in evidence["expert_demo_seeds"]]
    if not set(selected).issubset(set(rollout_train)):
        raise ValueError("expert_demo seeds must be a subset of rollout_train")
    if len(set(selected)) != len(selected):
        raise ValueError("expert_demo seeds must be unique")
    updated = json.loads(json.dumps(manifest))
    updated.setdefault("partitions", {})["expert_demo"] = selected
    updated["expert_demo_selection"] = {
        "source_partition": evidence.get("candidate_partition", DEFAULT_CANDIDATE_PARTITION),
        "rule": evidence["selection_rule"],
        "required_count": evidence["expert_demo_count"],
        "provisional": True,
        "evidence_path": None,
        "evidence_sha256": None,
    }
    feasibility = updated.setdefault("feasibility", {})
    feasibility.update(
        {
            "criterion": FEASIBILITY_CRITERION,
            "learning_policy_performance_consulted": False,
            "passed": False,
            "provisional_selection": True,
            "evidence_path": None,
            "evidence_sha256": None,
        }
    )
    updated["status"] = "candidate_unvalidated"
    return updated


def apply_expert_demo_to_manifest(manifest: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    errors = validate_feasibility_evidence(evidence)
    if errors:
        raise ValueError("; ".join(errors))
    rollout_train = manifest.get("partitions", {}).get("rollout_train", [])
    selected = [int(seed) for seed in evidence["expert_demo_seeds"]]
    if not set(selected).issubset(set(rollout_train)):
        raise ValueError("expert_demo seeds must be a subset of rollout_train")
    if len(set(selected)) != len(selected):
        raise ValueError("expert_demo seeds must be unique")
    updated = json.loads(json.dumps(manifest))
    updated.setdefault("partitions", {})["expert_demo"] = selected
    updated["expert_demo_selection"] = {
        "source_partition": evidence.get("candidate_partition", DEFAULT_CANDIDATE_PARTITION),
        "rule": evidence["selection_rule"],
        "required_count": evidence["expert_demo_count"],
        "evidence_path": None,
        "evidence_sha256": None,
    }
    feasibility = updated.setdefault("feasibility", {})
    feasibility.update(
        {
            "criterion": FEASIBILITY_CRITERION,
            "learning_policy_performance_consulted": False,
            "passed": bool(evidence["passed"]),
            "evidence_path": None,
            "evidence_sha256": None,
        }
    )
    return updated


def load_task_probe_args(task_name: str, task_config: str = DEFAULT_TASK_CONFIG) -> tuple[Any, dict[str, Any]]:
    import os
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    envs_module = importlib.import_module(f"envs.{task_name}")
    task_env = getattr(envs_module, task_name)()
    config_path = REPO_ROOT / "task_config" / f"{task_config}.yml"
    args = yaml.load(config_path.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["save_path"] = os.path.join(args.get("save_path", "./data"), task_name, task_config)

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    embodiment_types = yaml.load(Path(embodiment_config_path).read_text(encoding="utf-8"), Loader=yaml.FullLoader)

    def get_embodiment_file(emb_type: str) -> str:
        robot_file = embodiment_types[emb_type]["file_path"]
        if robot_file is None:
            raise RuntimeError("missing embodiment files")
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("number of embodiment config parameters should be 1 or 3")

    def get_embodiment_config(robot_file: str) -> dict[str, Any]:
        robot_config_file = Path(robot_file) / "config.yml"
        return yaml.load(robot_config_file.read_text(encoding="utf-8"), Loader=yaml.FullLoader)

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["need_plan"] = True
    args["render_freq"] = 0
    args["collect_data"] = False
    return task_env, args


def probe_seed_solvability(task_env: Any, args: dict[str, Any], seed: int, episode_idx: int = 0) -> dict[str, Any]:
    row = {"seed": int(seed), "episode_idx": int(episode_idx), "passed": False, "error_type": None, "error_message": None}
    try:
        task_env.setup_demo(now_ep_num=episode_idx, seed=seed, **args)
        task_env.play_once()
        if not (task_env.plan_success and task_env.check_success()):
            raise RuntimeError(f"Saved seed {seed} failed pre-motion validation for episode {episode_idx}")
        row["passed"] = True
    except Exception as exc:
        error_type, message = classify_probe_error(exc)
        row["error_type"] = error_type
        row["error_message"] = message
    finally:
        task_env.close_env()
        if args.get("render_freq"):
            viewer = getattr(task_env, "viewer", None)
            if viewer is not None:
                viewer.close()
    return row


def scan_candidate_seeds(
    task_name: str,
    candidate_seeds: list[int],
    *,
    task_config: str = DEFAULT_TASK_CONFIG,
    start_index: int = 0,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    task_env, args = load_task_probe_args(task_name, task_config)
    end_index = len(candidate_seeds) if limit is None else min(len(candidate_seeds), start_index + limit)
    results: list[dict[str, Any]] = []
    for index in range(start_index, end_index):
        seed = candidate_seeds[index]
        results.append(probe_seed_solvability(task_env, args, seed, episode_idx=0))
    return results
