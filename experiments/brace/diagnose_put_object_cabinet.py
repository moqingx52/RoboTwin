#!/usr/bin/env python3
"""Layered expert diagnostic for put_object_cabinet feasibility failures.

Reads archived feasibility evidence (with per-probe rows), tiers seeds by repeat
outcomes, and re-runs an instrumented expert probe that separates plan vs check
failures and records check_success sub-conditions.

Does not mutate manifests or expand seed pools.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.multitask_protocol import file_sha256
from experiments.brace.replay_audit import read_json, write_json_atomic
from experiments.brace.seed_feasibility import load_task_probe_args


PLAY_ONCE_PHASES = (
    "grasp_object",
    "grasp_drawer",
    "pull_drawer",
    "lift_object",
    "place_object",
)


def tier_from_probes(row: dict[str, Any]) -> str:
    probes = row.get("probes")
    if isinstance(probes, list) and probes:
        passed_count = sum(1 for probe in probes if probe.get("passed"))
        total = len(probes)
    else:
        passed_count = int(row.get("probe_passed_count") or 0)
        total = int(row.get("probe_repeats") or 1)
    if passed_count >= total:
        return "stable_pass"
    if passed_count == 0:
        return "stable_fail"
    return "flaky"


def check_success_components(task_env: Any) -> dict[str, Any]:
    object_pose = task_env.object.get_pose().p
    target_pose = task_env.cabinet.get_functional_point(0)
    dx = float(abs(object_pose[0] - target_pose[0]))
    dy = float(abs(object_pose[1] - target_pose[1]))
    xy_ok = bool(dx < 0.05 and dy < 0.05)
    dz = float(object_pose[2] - task_env.origin_z)
    dz_ok = bool(dz > 0.007 and dz < 0.12)
    arm_tag = getattr(task_env, "arm_tag", None)
    if arm_tag is not None and str(arm_tag) == "left":
        gripper_open = bool(task_env.robot.is_left_gripper_open())
    elif arm_tag is not None:
        gripper_open = bool(task_env.robot.is_right_gripper_open())
    else:
        gripper_open = None
    return {
        "dx": dx,
        "dy": dy,
        "xy_ok": xy_ok,
        "dz": dz,
        "dz_ok": dz_ok,
        "gripper_open": gripper_open,
        "check_success": bool(xy_ok and dz_ok and gripper_open),
    }


def run_phased_play(args: dict[str, Any], seed: int, episode_idx: int) -> dict[str, Any]:
    from envs.utils import ArmTag

    import importlib

    task_env = getattr(importlib.import_module("envs.put_object_cabinet"), "put_object_cabinet")()
    row: dict[str, Any] = {
        "seed": int(seed),
        "episode_idx": int(episode_idx),
        "plan_success": False,
        "check_success": False,
        "error_type": None,
        "error_message": None,
        "failed_phase": None,
        "object_model": None,
        "object_model_id": None,
        "arm_tag": None,
        "opposite_arm_tag": None,
        "check_components": None,
    }
    try:
        task_env.setup_demo(now_ep_num=episode_idx, seed=seed, **args)
        row["object_model"] = getattr(task_env, "selected_modelname", None)
        row["object_model_id"] = int(getattr(task_env, "selected_model_id", -1))

        arm_tag = ArmTag("right" if task_env.object.get_pose().p[0] > 0 else "left")
        task_env.arm_tag = arm_tag
        task_env.origin_z = task_env.object.get_pose().p[2]
        row["arm_tag"] = str(arm_tag)
        row["opposite_arm_tag"] = str(arm_tag.opposite)

        phases = [
            ("grasp_object", lambda: task_env.move(task_env.grasp_actor(task_env.object, arm_tag=arm_tag, pre_grasp_dis=0.1))),
            (
                "grasp_drawer",
                lambda: task_env.move(task_env.grasp_actor(task_env.cabinet, arm_tag=arm_tag.opposite, pre_grasp_dis=0.05)),
            ),
        ]
        for phase_name, action in phases:
            action()
            if not task_env.plan_success:
                row["failed_phase"] = phase_name
                row["error_type"] = "expert_plan_failed"
                row["error_message"] = f"plan failed during {phase_name}"
                return row

        for _ in range(4):
            task_env.move(task_env.move_by_displacement(arm_tag=arm_tag.opposite, y=-0.04))
            if not task_env.plan_success:
                row["failed_phase"] = "pull_drawer"
                row["error_type"] = "expert_plan_failed"
                row["error_message"] = "plan failed during pull_drawer"
                return row

        task_env.move(task_env.move_by_displacement(arm_tag=arm_tag, z=0.15))
        if not task_env.plan_success:
            row["failed_phase"] = "lift_object"
            row["error_type"] = "expert_plan_failed"
            row["error_message"] = "plan failed during lift_object"
            return row

        target_pose = task_env.cabinet.get_functional_point(0)
        task_env.move(
            task_env.place_actor(
                task_env.object,
                arm_tag=arm_tag,
                target_pose=target_pose,
                pre_dis=0.13,
                dis=0.1,
            )
        )
        if not task_env.plan_success:
            row["failed_phase"] = "place_object"
            row["error_type"] = "expert_plan_failed"
            row["error_message"] = "plan failed during place_object"
            return row

        row["plan_success"] = True
        components = check_success_components(task_env)
        row["check_components"] = components
        row["check_success"] = bool(components["check_success"])
        if not row["check_success"]:
            row["failed_phase"] = "check_success"
            row["error_type"] = "expert_check_failed"
            row["error_message"] = (
                f"check failed dx={components['dx']:.4f} dy={components['dy']:.4f} "
                f"dz={components['dz']:.4f} gripper_open={components['gripper_open']}"
            )
        return row
    except Exception as exc:
        row["error_type"] = "expert_probe_error"
        row["error_message"] = str(exc).strip() or exc.__class__.__name__
        if row["failed_phase"] is None:
            row["failed_phase"] = "exception"
        return row
    finally:
        task_env.close_env()
        if args.get("render_freq"):
            viewer = getattr(task_env, "viewer", None)
            if viewer is not None:
                viewer.close()


def build_diagnostic_report(
    evidence: dict[str, Any],
    evidence_path: Path,
    *,
    tier_limits: dict[str, int],
    episode_idx: int = 0,
) -> dict[str, Any]:
    _task_env, args = load_task_probe_args("put_object_cabinet", evidence.get("task_config", "demo_clean"))
    tiers: dict[str, list[int]] = {"stable_pass": [], "flaky": [], "stable_fail": []}
    for row in evidence.get("results", []):
        tier = tier_from_probes(row)
        tiers[tier].append(int(row["seed"]))

    samples: dict[str, list[dict[str, Any]]] = {}
    for tier, seeds in tiers.items():
        limit = tier_limits.get(tier, len(seeds))
        samples[tier] = []
        for seed in seeds[:limit]:
            samples[tier].append(run_phased_play(args, seed, episode_idx))

    return {
        "schema_version": 1,
        "task": "put_object_cabinet",
        "stage": "expert_diagnostic",
        "source_evidence": {
            "path": str(evidence_path.resolve().relative_to(REPO_ROOT.resolve()))
            if evidence_path.resolve().is_relative_to(REPO_ROOT.resolve())
            else str(evidence_path),
            "sha256": file_sha256(evidence_path),
        },
        "tier_counts": {tier: len(seeds) for tier, seeds in tiers.items()},
        "tier_sample_limits": tier_limits,
        "play_once_phases": list(PLAY_ONCE_PHASES),
        "notes": [
            "Diagnostic only; does not mutate seed manifests or expand candidate pools.",
            "check_components.dz_ok enforces 0.007 < dz < 0.12 after lift_object z=0.15.",
        ],
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=BRACE_DIR / "archive" / "seed_feasibility_20260809" / "put_object_cabinet_supplemented_feasibility.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stable-pass-limit", type=int, default=5)
    parser.add_argument("--flaky-limit", type=int, default=10)
    parser.add_argument("--stable-fail-limit", type=int, default=10)
    args = parser.parse_args()

    evidence_path = args.evidence.resolve()
    evidence = read_json(evidence_path)
    if evidence.get("task") != "put_object_cabinet":
        raise SystemExit(f"expected put_object_cabinet evidence, got {evidence.get('task')}")

    report = build_diagnostic_report(
        evidence,
        evidence_path,
        tier_limits={
            "stable_pass": args.stable_pass_limit,
            "flaky": args.flaky_limit,
            "stable_fail": args.stable_fail_limit,
        },
    )
    output = args.output or evidence_path.with_name("put_object_cabinet_expert_diagnostic.json")
    write_json_atomic(output, report)
    print(json.dumps({"output": str(output), "tier_counts": report["tier_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
