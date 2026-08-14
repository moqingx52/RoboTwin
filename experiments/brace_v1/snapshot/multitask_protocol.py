#!/usr/bin/env python3
"""Validate and materialize the BRACE RoboTwin multitask preregistration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.replay_audit import read_json, write_json_atomic


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def sha256_sidecar_valid(path: Path) -> bool:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        return False
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    return bool(fields) and fields[0] == file_sha256(path)


SUPPLEMENT_PARTITION_NAME = "expert_demo_supplement"
SUPPLEMENT_SELECTION_RULE = "original_pool_first_then_supplement_ascending"


def base_partition_counts(protocol: dict[str, Any]) -> list[tuple[str, int]]:
    cfg = protocol.get("seed_partitions", {})
    return [
        ("rollout_train", int(cfg.get("rollout_train_count", 100))),
        ("anchor_candidate", int(cfg.get("anchor_candidate_count", 200))),
        ("census_candidate", int(cfg.get("census_candidate_count", 200))),
        ("confirm_easy", int(cfg.get("confirm_easy_count", 100))),
        ("confirm_hard", int(cfg.get("confirm_hard_count", 100))),
    ]


def validate_multitask_amendment(amendment_path: Path, protocol_path: Path) -> list[str]:
    """Validate a brace.multitask.v1.1 supplement amendment against its protocol."""
    amendment_path = amendment_path.resolve()
    protocol_path = protocol_path.resolve()
    if not amendment_path.is_file():
        return [f"missing amendment: {amendment_path}"]
    if not sha256_sidecar_valid(amendment_path):
        return [f"missing or invalid amendment SHA256 sidecar: {amendment_path}"]
    amendment = read_json(amendment_path)
    errors: list[str] = []
    if amendment.get("amendment_revision") != "brace.multitask.v1.1":
        errors.append("unexpected amendment_revision")
    applies_to = amendment.get("applies_to", {})
    if applies_to.get("protocol_revision") != "brace.multitask.v1":
        errors.append("amendment applies_to.protocol_revision must be brace.multitask.v1")
    if not protocol_path.is_file():
        errors.append(f"missing protocol: {protocol_path}")
        return errors
    if not sha256_sidecar_valid(protocol_path):
        errors.append(f"missing or invalid protocol SHA256 sidecar: {protocol_path}")
    if applies_to.get("protocol_sha256") != file_sha256(protocol_path):
        errors.append("amendment applies_to.protocol_sha256 does not match the protocol file")
    if not amendment.get("date") or not amendment.get("reason"):
        errors.append("amendment must record date and reason")
    original = amendment.get("original_evidence", [])
    if not isinstance(original, list) or not original:
        errors.append("amendment must bind original_evidence (task + evidence_sha256)")
    for entry in original:
        if not isinstance(entry, dict) or not entry.get("task") or not entry.get("evidence_sha256"):
            errors.append("original_evidence entries need task and evidence_sha256")
    supplement = amendment.get("supplement", {})
    if supplement.get("partition_name") != SUPPLEMENT_PARTITION_NAME:
        errors.append(f"supplement.partition_name must be {SUPPLEMENT_PARTITION_NAME}")
    if supplement.get("selection_rule") != SUPPLEMENT_SELECTION_RULE:
        errors.append(f"supplement.selection_rule must be {SUPPLEMENT_SELECTION_RULE}")
    if supplement.get("uniform_for_all_tasks") is not True:
        errors.append("supplement must be uniform for all tasks")
    if supplement.get("criterion") != "expert_script_simulator_solvability_only":
        errors.append("supplement criterion must be expert-script solvability only")
    if supplement.get("learned_policy_performance_consulted") is not False:
        errors.append("supplement must not consult learned-policy performance")
    try:
        offset = int(supplement.get("task_base_offset", -1))
        count = int(supplement.get("count", -1))
    except (TypeError, ValueError):
        offset, count = -1, -1
    if offset < 0 or count < 1:
        errors.append("supplement task_base_offset/count must be non-negative integers")
    elif protocol_path.is_file():
        protocol = read_json(protocol_path)
        stride = int(protocol.get("seed_partitions", {}).get("partition_stride", 0))
        base_total = sum(count for _, count in base_partition_counts(protocol))
        if offset != base_total:
            errors.append(
                f"supplement offset {offset} must equal the total base partition count {base_total}"
            )
        if offset + count > stride:
            errors.append(
                f"supplement range [{offset},{offset + count}) exceeds partition_stride {stride}"
            )
    return errors


def validate_method_pilot_summary(summary: dict[str, Any], gate: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if summary.get("task") != gate.get("task"):
        errors.append(f"task must be {gate.get('task')}")
    if summary.get("stage") != gate.get("stage"):
        errors.append(f"stage must be {gate.get('stage')}")
    if summary.get("status") != "passed" or summary.get("complete") is not True:
        errors.append("pilot must be complete with status=passed")
    result = summary.get("preservation_go_no_go", {})
    if not isinstance(result, dict):
        return errors + ["preservation_go_no_go must be an object"]
    if result.get("passed") is not True:
        errors.append("preservation_go_no_go must pass")
    required_seeds = gate.get("paired_seed_count", 5)
    if result.get("paired_seed_count") != required_seeds:
        errors.append(f"preservation_go_no_go requires exactly {required_seeds} paired seeds")
    effect = result.get("effect_point_estimate")
    threshold = gate.get("effect_point_estimate_gt", 0.0)
    if not isinstance(effect, (int, float)) or isinstance(effect, bool) or effect <= threshold:
        errors.append("preservation effect point estimate does not exceed the preregistered threshold")
    wins = result.get("directional_wins")
    minimum_wins = gate.get("minimum_directional_wins", 4)
    if not isinstance(wins, int) or isinstance(wins, bool) or wins < minimum_wins:
        errors.append("preservation directional wins are below the preregistered threshold")
    return errors


def validate_multitask_protocol(
    protocol_path: Path,
    *,
    require_method_freeze: bool = False,
) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    protocol = read_json(protocol_path)
    errors: list[str] = []
    if protocol.get("protocol_revision") != "brace.multitask.v1":
        errors.append("unexpected protocol_revision")
    if bool(protocol.get("exploratory", True)):
        errors.append("multitask protocol must be non-exploratory")
    development_gate = protocol.get("method_development_gate", {})
    if development_gate != {
        "task": "place_container_plate",
        "stage": "brace_v2_intervention_pilot",
        "paired_seed_count": 5,
        "effect_point_estimate_gt": 0.0,
        "minimum_directional_wins": 4,
    }:
        errors.append("unexpected or incomplete method_development_gate")

    task_path = resolve_repo_path(protocol.get("task_manifest", ""))
    leaderboard_path = resolve_repo_path(protocol.get("leaderboard_snapshot", ""))
    for label, path in (("task manifest", task_path), ("leaderboard snapshot", leaderboard_path)):
        if not path.is_file():
            errors.append(f"missing {label}: {path}")
        elif not sha256_sidecar_valid(path):
            errors.append(f"missing or invalid {label} SHA256 sidecar: {path}")
    if protocol_path.is_file() and not sha256_sidecar_valid(protocol_path):
        errors.append(f"missing or invalid protocol SHA256 sidecar: {protocol_path}")
    amendment_path = resolve_repo_path(protocol.get("amendment", "")) if protocol.get("amendment") else None
    if amendment_path is not None:
        errors.extend(f"amendment: {error}" for error in validate_multitask_amendment(amendment_path, protocol_path))
    if errors:
        return {"passed": False, "errors": errors}

    tasks = read_json(task_path)
    leaderboard = read_json(leaderboard_path)
    development = [str(task) for task in tasks.get("development_tasks", [])]
    heldout = [str(task) for task in tasks.get("heldout_tasks", [])]
    all_tasks = development + heldout
    if tasks.get("status") != "frozen":
        errors.append("task manifest is not frozen")
    if len(development) != 2 or len(heldout) != 10 or len(set(all_tasks)) != 12:
        errors.append("task panel must contain 2 development and 10 distinct held-out tasks")
    if set(tasks.get("tasks", {})) != set(all_tasks):
        errors.append("task metadata does not match development + held-out task lists")
    if set(leaderboard.get("results", {})) != set(all_tasks):
        errors.append("leaderboard snapshot does not exactly cover the task panel")
    if int(protocol.get("inference", {}).get("heldout_task_count", -1)) != len(heldout):
        errors.append("protocol heldout_task_count does not match task manifest")

    for task in all_tasks:
        if not (REPO_ROOT / "envs" / f"{task}.py").is_file():
            errors.append(f"missing task environment: {task}")
        if not (REPO_ROOT / "description" / "task_instruction" / f"{task}.json").is_file():
            errors.append(f"missing task instruction: {task}")
        result = leaderboard.get("results", {}).get(task, {})
        if set(result) != {"RDT", "Pi0", "ACT", "DP", "DP3"}:
            errors.append(f"incomplete leaderboard policies for {task}")
        for policy, pair in result.items():
            if not isinstance(pair, list) or len(pair) != 2 or any(not 0 <= float(value) <= 1 for value in pair):
                errors.append(f"invalid leaderboard result for {task}/{policy}")
        expected_dp = tasks.get("tasks", {}).get(task, {}).get("leaderboard_dp_easy")
        actual_dp = result.get("DP", [None])[0]
        if expected_dp is None or actual_dp is None or abs(float(expected_dp) - float(actual_dp)) > 1e-12:
            errors.append(f"DP Easy mismatch for {task}")

    method_path = resolve_repo_path(protocol.get("method_freeze", ""))
    method_status = "missing"
    method_sha256 = None
    if method_path.is_file():
        method = read_json(method_path)
        method_status = str(method.get("status", "unknown"))
        method_sha256 = file_sha256(method_path)
        if require_method_freeze:
            if not bool(method.get("frozen")) or method_status != "frozen":
                errors.append("method freeze is not frozen")
            source = resolve_repo_path(method.get("source_run", "")) if method.get("source_run") else None
            if source is None or not source.exists():
                errors.append("method freeze source_run is missing")
            source_summary = source / "summary.json" if source and source.is_dir() else source
            expected_sha = method.get("source_summary_sha256")
            if source_summary is None or not source_summary.is_file() or file_sha256(source_summary) != expected_sha:
                errors.append("method freeze source summary SHA mismatch")
            else:
                source_payload = read_json(source_summary)
                gate = protocol.get("method_development_gate", {})
                errors.extend(f"method freeze source: {error}" for error in validate_method_pilot_summary(source_payload, gate))
                if method.get("source_pilot_gate") != source_payload.get("preservation_go_no_go"):
                    errors.append("method freeze source_pilot_gate mismatch")
            config_value = method.get("method_config_path")
            config_path = resolve_repo_path(config_value) if config_value else None
            if config_path is None or not config_path.is_file():
                errors.append("method freeze method config is missing")
            elif file_sha256(config_path) != method.get("method_config_sha256"):
                errors.append("method freeze method config SHA mismatch")
    elif require_method_freeze:
        errors.append(f"missing method freeze: {method_path}")

    return {
        "schema_version": 1,
        "stage": "multitask_protocol_validation",
        "passed": not errors,
        "errors": errors,
        "protocol_path": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "task_manifest_path": str(task_path),
        "task_manifest_sha256": file_sha256(task_path),
        "leaderboard_snapshot_path": str(leaderboard_path),
        "leaderboard_snapshot_sha256": file_sha256(leaderboard_path),
        "method_freeze_path": str(method_path),
        "method_freeze_status": method_status,
        "method_freeze_sha256": method_sha256,
        "development_tasks": development,
        "heldout_tasks": heldout,
    }


def build_seed_manifest(
    protocol: dict[str, Any],
    task_manifest: dict[str, Any],
    task: str,
    *,
    amendment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    all_tasks = list(task_manifest["development_tasks"]) + list(task_manifest["heldout_tasks"])
    if task not in all_tasks:
        raise ValueError(f"task is not preregistered: {task}")
    cfg = protocol["seed_partitions"]
    task_index = all_tasks.index(task)
    cursor = int(cfg["start_seed"]) + task_index * int(cfg["partition_stride"])
    counts = base_partition_counts(protocol)
    partitions: dict[str, list[int]] = {}
    for name, count in counts:
        partitions[name] = list(range(cursor, cursor + count))
        cursor += count
    supplement = None
    if amendment is not None:
        supplement = amendment.get("supplement", {})
    elif protocol.get("amendment"):
        amendment_path = resolve_repo_path(protocol["amendment"])
        if amendment_path.is_file():
            supplement = read_json(amendment_path).get("supplement", {})
    if supplement:
        name = str(supplement.get("partition_name", SUPPLEMENT_PARTITION_NAME))
        offset = int(supplement.get("task_base_offset", 0))
        count = int(supplement.get("count", 0))
        base = int(cfg["start_seed"]) + task_index * int(cfg["partition_stride"])
        partitions[name] = list(range(base + offset, base + offset + count))
    flat = [seed for values in partitions.values() for seed in values]
    if len(flat) != len(set(flat)):
        raise AssertionError("generated seed partitions overlap")
    notes = [
        "Candidate env seeds are deterministic and disjoint.",
        "rollout_train retains the full 100-seed candidate pool; expert_demo is selected later.",
        "partitions hold only mutually exclusive candidate/eval partitions; expert_demo lives in cohorts.",
        "Freeze only from expert-script/simulator solvability evidence; learned-policy outcomes must not be consulted.",
        "Set status=frozen only after recording passed feasibility evidence and its SHA256.",
    ]
    if supplement:
        notes.append(
            f"{SUPPLEMENT_PARTITION_NAME} is the amendment-supplied backup pool "
            "(brace.multitask.v1.1); used only when the original pool is insufficient."
        )
    return {
        "schema_version": 1,
        "task": task,
        "status": "candidate_unvalidated",
        "task_role": task_manifest["tasks"][task]["role"],
        "feasibility": {
            "criterion": "expert_script_simulator_solvability_only",
            "learning_policy_performance_consulted": False,
            "evidence_path": None,
            "evidence_sha256": None,
            "passed": False,
        },
        "expert_demo_selection": {
            "source_partition": "rollout_train",
            "rule": "first_n_solvable_in_manifest_order",
            "required_count": int(task_manifest.get("expert_demonstrations_per_task", 50)),
            "evidence_path": None,
            "evidence_sha256": None,
        },
        "cohorts": {
            "expert_demo": [],
        },
        "partitions": partitions,
        "policy_seed_offsets": cfg["policy_seed_offsets"],
        "notes": notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "generate-seeds"))
    parser.add_argument("--protocol", type=Path, default=BRACE_DIR / "multitask_protocol.v1.json")
    parser.add_argument("--require-method-freeze", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=BRACE_DIR / "seeds" / "multitask_v1")
    parser.add_argument("--tasks", nargs="*")
    args = parser.parse_args()

    validation = validate_multitask_protocol(args.protocol, require_method_freeze=args.require_method_freeze)
    if args.command == "validate":
        if args.output:
            write_json_atomic(args.output, validation)
        print(json.dumps(validation, indent=2))
        return 0 if validation["passed"] else 2
    if not validation["passed"]:
        raise SystemExit("cannot generate seeds for an invalid multitask protocol")

    protocol = read_json(args.protocol.resolve())
    task_manifest = read_json(resolve_repo_path(protocol["task_manifest"]))
    selected = args.tasks or task_manifest["development_tasks"] + task_manifest["heldout_tasks"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task in selected:
        output = args.output_dir / f"{task}.json"
        write_json_atomic(output, build_seed_manifest(protocol, task_manifest, task))
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
