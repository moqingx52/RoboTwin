#!/usr/bin/env python3
"""Phase-level diagnostic for handover_mic expert/check instability."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
MANIFEST_PATH = BRACE_DIR / "seeds" / "multitask_v1" / "handover_mic.json"
DEFAULT_SEEDS = (140004, 140005, 140007, 140018, 140000)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.multitask_protocol import file_sha256
from experiments.brace.replay_audit import write_json_atomic
from experiments.brace.seed_feasibility import load_task_probe_args


PHASES = (
    "grasp_microphone",
    "raise_giver",
    "move_to_middle",
    "receiver_grasp",
    "giver_open",
    "final_separation",
)


def resolve_seed_cases(manifest: dict[str, Any], seeds: list[int]) -> list[dict[str, int]]:
    cohort = [int(seed) for seed in manifest.get("cohorts", {}).get("expert_demo", [])]
    cases: list[dict[str, int]] = []
    for seed in seeds:
        if seed not in cohort:
            raise ValueError(f"seed {seed} is not in handover_mic frozen expert_demo cohort")
        cases.append({"seed": int(seed), "episode_idx": cohort.index(seed)})
    return cases


def vec(values: Any) -> list[float]:
    return [float(value) for value in np.asarray(values).reshape(-1)]


def gripper_state(task_env: Any, arm: Any) -> dict[str, bool]:
    if str(arm) == "left":
        return {
            "open": bool(task_env.is_left_gripper_open()),
            "closed": bool(task_env.is_left_gripper_close()),
        }
    return {
        "open": bool(task_env.is_right_gripper_open()),
        "closed": bool(task_env.is_right_gripper_close()),
    }


def check_components(task_env: Any, giver: Any, receiver: Any) -> dict[str, Any]:
    microphone_fp = np.asarray(task_env.microphone.get_functional_point(0), dtype=float)
    microphone_pose = np.asarray(task_env.microphone.get_pose().p, dtype=float)
    contacts = task_env.get_gripper_actor_contact_position("018_microphone")
    giver_tcp = np.asarray(
        task_env.robot.get_left_tcp_pose()[:3] if str(giver) == "left" else task_env.robot.get_right_tcp_pose()[:3],
        dtype=float,
    )
    receiver_tcp = np.asarray(
        task_env.robot.get_left_tcp_pose()[:3] if str(receiver) == "left" else task_env.robot.get_right_tcp_pose()[:3],
        dtype=float,
    )
    giver_state = gripper_state(task_env, giver)
    receiver_state = gripper_state(task_env, receiver)
    height_ok = bool(microphone_fp[2] > 0.92)
    side_ok = bool(microphone_fp[0] < 0) if str(receiver) == "left" else bool(microphone_fp[0] > 0)
    contact_ok = bool(contacts)
    return {
        "microphone_functional_point": vec(microphone_fp),
        "microphone_actor_position": vec(microphone_pose),
        "giver_tcp_position": vec(giver_tcp),
        "receiver_tcp_position": vec(receiver_tcp),
        "giver_tcp_distance_to_microphone": float(np.linalg.norm(giver_tcp - microphone_pose)),
        "receiver_tcp_distance_to_microphone": float(np.linalg.norm(receiver_tcp - microphone_pose)),
        "contact_count": len(contacts),
        "contact_positions": [vec(position) for position in contacts],
        "contact_ok": contact_ok,
        "giver_gripper": giver_state,
        "receiver_gripper": receiver_state,
        "height_ok": height_ok,
        "side_ok": side_ok,
        "check_success": bool(contact_ok and receiver_state["closed"] and giver_state["open"] and height_ok and side_ok),
    }


def run_one(args: dict[str, Any], *, seed: int, episode_idx: int, repeat: int) -> dict[str, Any]:
    from envs.handover_mic import handover_mic
    from envs._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
    from envs.utils import ArmTag

    task_env = handover_mic()
    row: dict[str, Any] = {
        "seed": seed,
        "episode_idx": episode_idx,
        "repeat": repeat,
        "microphone_model_id": None,
        "giver_arm": None,
        "receiver_arm": None,
        "phases": [],
        "plan_success": False,
        "check_success": False,
        "failed_phase": None,
        "error_type": None,
        "error_message": None,
    }
    try:
        task_env.setup_demo(now_ep_num=episode_idx, seed=seed, **args)
        giver = ArmTag("right" if task_env.microphone.get_pose().p[0] > 0 else "left")
        receiver = giver.opposite
        row["microphone_model_id"] = int(task_env.microphone_id)
        row["giver_arm"] = str(giver)
        row["receiver_arm"] = str(receiver)

        actions: list[tuple[str, Callable[[], Any]]] = [
            (
                "grasp_microphone",
                lambda: task_env.move(
                    task_env.grasp_actor(
                        task_env.microphone,
                        arm_tag=giver,
                        contact_point_id=[1, 9, 10, 11, 12, 13, 14, 15],
                        pre_grasp_dis=0.1,
                    )
                ),
            ),
            (
                "raise_giver",
                lambda: task_env.move(
                    task_env.move_by_displacement(
                        giver,
                        z=0.12,
                        quat=GRASP_DIRECTION_DIC["front_right" if giver == "left" else "front_left"],
                        move_axis="arm",
                    )
                ),
            ),
            (
                "move_to_middle",
                lambda: task_env.move(
                    task_env.place_actor(
                        task_env.microphone,
                        arm_tag=giver,
                        target_pose=task_env.handover_middle_pose,
                        functional_point_id=0,
                        pre_dis=0.0,
                        dis=0.0,
                        is_open=False,
                        constrain="free",
                    )
                ),
            ),
            (
                "receiver_grasp",
                lambda: task_env.move(
                    task_env.grasp_actor(
                        task_env.microphone,
                        arm_tag=receiver,
                        contact_point_id=[0, 2, 3, 4, 5, 6, 7, 8],
                        pre_grasp_dis=0.1,
                    )
                ),
            ),
            ("giver_open", lambda: task_env.move(task_env.open_gripper(giver))),
            (
                "final_separation",
                lambda: task_env.move(
                    task_env.move_by_displacement(giver, z=0.07, move_axis="arm"),
                    task_env.move_by_displacement(receiver, x=0.05 if receiver == "right" else -0.05),
                ),
            ),
        ]
        for phase_name, action in actions:
            action()
            snapshot = check_components(task_env, giver, receiver)
            row["phases"].append({"phase": phase_name, "plan_success": bool(task_env.plan_success), "state": snapshot})
            if not task_env.plan_success:
                row["failed_phase"] = phase_name
                row["error_type"] = "expert_plan_failed"
                row["error_message"] = f"plan failed during {phase_name}"
                return row

        row["plan_success"] = True
        final = row["phases"][-1]["state"]
        row["check_success"] = bool(final["check_success"])
        if not row["check_success"]:
            row["failed_phase"] = "check_success"
            row["error_type"] = "expert_check_failed"
            failed = [
                name
                for name, passed in (
                    ("contact", final["contact_ok"]),
                    ("receiver_closed", final["receiver_gripper"]["closed"]),
                    ("giver_open", final["giver_gripper"]["open"]),
                    ("height", final["height_ok"]),
                    ("side", final["side_ok"]),
                )
                if not passed
            ]
            row["error_message"] = "failed check components: " + ",".join(failed)
        return row
    except Exception as exc:
        row["failed_phase"] = row["failed_phase"] or "exception"
        row["error_type"] = row["error_type"] or type(exc).__name__
        row["error_message"] = row["error_message"] or (str(exc).strip() or type(exc).__name__)
        return row
    finally:
        try:
            task_env.close_env()
        except Exception as exc:
            row["error_type"] = row["error_type"] or "close_env_failed"
            row["error_message"] = row["error_message"] or (str(exc).strip() or type(exc).__name__)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed: dict[str, Any] = {}
    failed_components: dict[str, int] = {}
    for row in rows:
        seed = str(row["seed"])
        bucket = by_seed.setdefault(seed, {"attempts": 0, "passed": 0, "plan_failed": 0, "check_failed": 0})
        bucket["attempts"] += 1
        if row["plan_success"] and row["check_success"]:
            bucket["passed"] += 1
        elif row["error_type"] == "expert_plan_failed":
            bucket["plan_failed"] += 1
        elif row["error_type"] == "expert_check_failed":
            bucket["check_failed"] += 1
            final = row["phases"][-1]["state"] if row["phases"] else {}
            checks = {
                "contact": final.get("contact_ok"),
                "receiver_closed": final.get("receiver_gripper", {}).get("closed"),
                "giver_open": final.get("giver_gripper", {}).get("open"),
                "height": final.get("height_ok"),
                "side": final.get("side_ok"),
            }
            for name, passed in checks.items():
                if passed is False:
                    failed_components[name] = failed_components.get(name, 0) + 1
    return {"by_seed": by_seed, "failed_check_components": failed_components}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--diagnostic-repeats", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.diagnostic_repeats < 1:
        raise SystemExit("diagnostic-repeats must be positive")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    cases = resolve_seed_cases(manifest, args.seeds)
    _task_env, probe_args = load_task_probe_args("handover_mic", "demo_clean")
    rows = [
        run_one(probe_args, seed=case["seed"], episode_idx=case["episode_idx"], repeat=repeat)
        for case in cases
        for repeat in range(args.diagnostic_repeats)
    ]
    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = BRACE_DIR / "runs" / f"handover_mic_diagnostic_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        output = run_dir / "diagnostic.json"
    elif output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    report = {
        "schema_version": 1,
        "stage": "handover_mic_expert_diagnostic",
        "diagnostic_only": True,
        "task": "handover_mic",
        "diagnostic_repeats": args.diagnostic_repeats,
        "phases": list(PHASES),
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "manifest_sha256": file_sha256(MANIFEST_PATH),
        "task_config_sha256": file_sha256(REPO_ROOT / "task_config" / "demo_clean.yml"),
        "cases": cases,
        "summary": summarize(rows),
        "rows": rows,
    }
    write_json_atomic(output, report)
    print(json.dumps({"output": str(output), "summary": report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
