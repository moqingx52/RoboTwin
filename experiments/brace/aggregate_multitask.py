#!/usr/bin/env python3
"""Aggregate preregistered BRACE multitask paired evaluation artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.multitask_protocol import (
    file_sha256,
    resolve_repo_path,
    sha256_sidecar_valid,
    validate_multitask_protocol,
)
from experiments.brace.replay_audit import read_json, write_json_atomic


FACTORIAL_ARMS = ("n1", "b1", "b2", "b3")
REQUIRED_ARMS = ("u0", *FACTORIAL_ARMS)


def rows_by_key(payload: dict[str, Any], split: str) -> dict[tuple[int, int], float]:
    rows: dict[tuple[int, int], float] = {}
    for row in payload.get("rows", []):
        if row.get("split") != split:
            continue
        key = (int(row["env_seed"]), int(row.get("repeat", 0)))
        if key in rows:
            raise ValueError(f"duplicate work item for split={split}: {key}")
        if "success" not in row or not isinstance(row["success"], (bool, np.bool_)):
            raise ValueError(f"missing or non-boolean success for split={split}: {key}")
        rows[key] = float(row["success"])
    return rows


def validate_eval_payload(
    payload: dict[str, Any],
    protocol: dict[str, Any],
    *,
    label: str,
    seed_manifest: dict[str, Any] | None = None,
) -> None:
    if not bool(payload.get("progress", {}).get("complete")):
        raise ValueError(f"incomplete eval artifact: {label}")
    errors: list[str] = []
    for split, cfg in protocol["evaluation"]["splits"].items():
        rows = rows_by_key(payload, split)
        expected_envs = int(cfg["env_seed_count"])
        expected_repeats = int(cfg["policy_repeats"])
        envs = {key[0] for key in rows}
        if len(envs) != expected_envs:
            errors.append(f"{split} env seeds={len(envs)}, expected {expected_envs}")
        if len(rows) != expected_envs * expected_repeats:
            errors.append(f"{split} work items={len(rows)}, expected {expected_envs * expected_repeats}")
        for env_seed in envs:
            repeats = {repeat for seed, repeat in rows if seed == env_seed}
            if repeats != set(range(expected_repeats)):
                errors.append(f"{split}/{env_seed} repeats differ")
                break
        partition = cfg.get("seed_manifest_partition")
        if partition:
            if seed_manifest is None:
                errors.append(f"{split} requires seed manifest partition {partition}")
            else:
                allowed = set(seed_manifest.get("partitions", {}).get(partition, []))
                if not allowed:
                    errors.append(f"seed manifest missing partition {partition}")
                elif not envs.issubset(allowed):
                    errors.append(f"{split} env seeds are outside partition {partition}")
    if errors:
        raise ValueError(f"invalid eval artifact {label}: " + "; ".join(errors))


def load_artifact_index(
    index_path: Path,
    *,
    tasks: list[str],
    training_seeds: list[int],
    protocol: dict[str, Any] | None = None,
    seed_manifests: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, dict[int, dict[str, Any]]]], dict[str, Any]]:
    index = read_json(index_path)
    matrix: dict[str, dict[str, dict[int, dict[str, Any]]]] = {
        task: {arm: {} for arm in ("base", *REQUIRED_ARMS)} for task in tasks
    }
    errors: list[str] = []
    for entry in index.get("artifacts", []):
        task = str(entry.get("task"))
        arm = str(entry.get("arm"))
        seed = int(entry.get("training_seed", 0))
        if task not in matrix or arm not in matrix[task]:
            errors.append(f"unexpected artifact identity: {task}/{arm}/s{seed}")
            continue
        expected_seeds = [0] if arm == "base" else training_seeds
        if seed not in expected_seeds or seed in matrix[task][arm]:
            errors.append(f"invalid or duplicate training seed: {task}/{arm}/s{seed}")
            continue
        path = resolve_repo_path(str(entry.get("path", "")))
        if not path.is_file():
            errors.append(f"missing artifact: {path}")
            continue
        actual_sha = file_sha256(path)
        if entry.get("sha256") != actual_sha:
            errors.append(f"artifact SHA mismatch: {path}")
            continue
        payload = read_json(path)
        if protocol is not None:
            try:
                validate_eval_payload(
                    payload,
                    protocol,
                    label=f"{task}/{arm}/s{seed}",
                    seed_manifest=seed_manifests.get(task) if seed_manifests else None,
                )
            except ValueError as exc:
                errors.append(str(exc))
                continue
        matrix[task][arm][seed] = payload
    for task in tasks:
        if set(matrix[task]["base"]) != {0}:
            errors.append(f"missing base artifact for {task}")
        for arm in REQUIRED_ARMS:
            if set(matrix[task][arm]) != set(training_seeds):
                errors.append(f"incomplete {task}/{arm} training seed matrix")
    if errors:
        raise ValueError("invalid multitask artifact index: " + "; ".join(errors[:20]))
    return matrix, index


def paired_env_values(
    left: dict[tuple[int, int], float],
    right: dict[tuple[int, int], float],
) -> dict[int, float]:
    if set(left) != set(right):
        raise ValueError(
            f"paired work items differ: left_only={len(set(left) - set(right))}, "
            f"right_only={len(set(right) - set(left))}"
        )
    env_seeds = sorted({key[0] for key in left})
    return {
        env_seed: float(np.mean([left[key] - right[key] for key in left if key[0] == env_seed]))
        for env_seed in env_seeds
    }


def mean_maps(*maps: dict[int, float]) -> dict[int, float]:
    if not maps:
        return {}
    keys = set(maps[0])
    if any(set(values) != keys for values in maps[1:]):
        raise ValueError("cannot average unpaired env-seed maps")
    return {key: float(np.mean([values[key] for values in maps])) for key in sorted(keys)}


def subtract_maps(left: dict[int, float], right: dict[int, float]) -> dict[int, float]:
    if set(left) != set(right):
        raise ValueError("cannot subtract unpaired env-seed maps")
    return {key: left[key] - right[key] for key in sorted(left)}


def effect_maps_for_task(
    task_payloads: dict[str, dict[int, dict[str, Any]]],
    training_seeds: list[int],
    *,
    preservation_split: str = "preservation",
    adaptation_split: str = "confirm_easy",
) -> dict[str, dict[int, dict[int, float]]]:
    effects = {
        "anchor_main_preservation": {},
        "credit_main_adaptation": {},
        "brace_preservation": {},
        "brace_adaptation": {},
        "interaction_adaptation": {},
    }
    for seed in training_seeds:
        pres = {arm: rows_by_key(task_payloads[arm][seed], preservation_split) for arm in FACTORIAL_ARMS}
        adapt = {arm: rows_by_key(task_payloads[arm][seed], adaptation_split) for arm in FACTORIAL_ARMS}
        if any(not values for values in (*pres.values(), *adapt.values())):
            raise ValueError(f"empty primary split for training seed {seed}")
        effects["anchor_main_preservation"][seed] = paired_env_values(
            {key: (pres["b2"][key] + pres["b3"][key]) / 2 for key in pres["b2"]},
            {key: (pres["n1"][key] + pres["b1"][key]) / 2 for key in pres["n1"]},
        )
        effects["credit_main_adaptation"][seed] = paired_env_values(
            {key: (adapt["b1"][key] + adapt["b3"][key]) / 2 for key in adapt["b1"]},
            {key: (adapt["n1"][key] + adapt["b2"][key]) / 2 for key in adapt["n1"]},
        )
        effects["brace_preservation"][seed] = paired_env_values(pres["b3"], pres["b1"])
        effects["brace_adaptation"][seed] = paired_env_values(adapt["b3"], adapt["b1"])
        b3_b2 = paired_env_values(adapt["b3"], adapt["b2"])
        b1_n1 = paired_env_values(adapt["b1"], adapt["n1"])
        effects["interaction_adaptation"][seed] = subtract_maps(b3_b2, b1_n1)
    return effects


def mean_effect(effect: dict[int, dict[int, float]]) -> float:
    return float(np.mean([np.mean(list(env_values.values())) for env_values in effect.values()]))


def hierarchical_task_bootstrap(
    effects: dict[str, dict[int, dict[int, float]]],
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    tasks = sorted(effects)
    if not tasks:
        raise ValueError("no tasks for bootstrap")
    point = float(np.mean([mean_effect(effects[task]) for task in tasks]))
    samples: list[float] = []
    for _ in range(replicates):
        task_draw: list[float] = []
        for _ in tasks:
            task = str(rng.choice(tasks))
            training_seeds = sorted(effects[task])
            seed_draw: list[float] = []
            for _ in training_seeds:
                training_seed = int(rng.choice(training_seeds))
                env_map = effects[task][training_seed]
                env_seeds = sorted(env_map)
                seed_draw.append(float(np.mean([env_map[int(rng.choice(env_seeds))] for _ in env_seeds])))
            task_draw.append(float(np.mean(seed_draw)))
        samples.append(float(np.mean(task_draw)))
    alpha = (1 - confidence_level) / 2
    return {
        "point": point,
        "lower": float(np.quantile(samples, alpha)),
        "upper": float(np.quantile(samples, 1 - alpha)),
        "replicates": replicates,
    }


def aggregate_multitask(
    protocol: dict[str, Any],
    task_manifest: dict[str, Any],
    matrix: dict[str, dict[str, dict[int, dict[str, Any]]]],
    *,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    heldout = [str(task) for task in task_manifest["heldout_tasks"]]
    training_seeds = [int(seed) for seed in protocol["training"]["training_seeds"]]
    effect_names = (
        "anchor_main_preservation",
        "credit_main_adaptation",
        "brace_preservation",
        "brace_adaptation",
        "interaction_adaptation",
    )
    task_effect_maps: dict[str, dict[str, dict[int, dict[int, float]]]] = {}
    task_summary: dict[str, Any] = {}
    for task in heldout:
        effects = effect_maps_for_task(matrix[task], training_seeds)
        task_effect_maps[task] = effects
        task_summary[task] = {name: mean_effect(effects[name]) for name in effect_names}
        base_easy = rows_by_key(matrix[task]["base"][0], "confirm_easy")
        base_hard = rows_by_key(matrix[task]["base"][0], "confirm_hard")
        base_preservation = rows_by_key(matrix[task]["base"][0], "preservation")
        b3_easy = [rows_by_key(matrix[task]["b3"][seed], "confirm_easy") for seed in training_seeds]
        b3_hard = [rows_by_key(matrix[task]["b3"][seed], "confirm_hard") for seed in training_seeds]
        b3_preservation = [rows_by_key(matrix[task]["b3"][seed], "preservation") for seed in training_seeds]
        if any(set(values) != set(base_preservation) for values in b3_preservation):
            raise ValueError(f"B3/base preservation keys differ for {task}")
        task_summary[task]["local_base_easy"] = float(np.mean(list(base_easy.values())))
        task_summary[task]["local_base_hard"] = float(np.mean(list(base_hard.values())))
        task_summary[task]["b3_easy"] = float(np.mean([np.mean(list(values.values())) for values in b3_easy]))
        task_summary[task]["b3_hard"] = float(np.mean([np.mean(list(values.values())) for values in b3_hard]))
        task_summary[task]["b3_minus_fresh_base_retention"] = float(
            np.mean([
                np.mean([values[key] - base_preservation[key] for key in values])
                for values in b3_preservation
            ])
        )
        official_dp_easy = float(task_manifest["tasks"][task]["leaderboard_dp_easy"])
        task_summary[task]["official_dp_easy"] = official_dp_easy
        task_summary[task]["local_dp_reproduction_deviation"] = (
            task_summary[task]["local_base_easy"] - official_dp_easy
        )
        reproduction_limit = float(
            protocol["launch_gates"].get("local_dp_reproduction_max_absolute_deviation", 0.10)
        )
        task_summary[task]["local_dp_reproduction_deviation_gt_10pp"] = (
            abs(task_summary[task]["local_dp_reproduction_deviation"]) > reproduction_limit
        )

    inference = protocol["inference"]
    pooled: dict[str, Any] = {}
    for offset, name in enumerate(effect_names):
        pooled[name] = hierarchical_task_bootstrap(
            {task: task_effect_maps[task][name] for task in heldout},
            replicates=int(inference["bootstrap_replicates"]),
            seed=int(inference["bootstrap_seed"]) + offset,
            confidence_level=float(inference["confidence_level"]),
        )
    positive_tasks = sum(task_summary[task]["brace_preservation"] > 0 for task in heldout)
    severe = [
        task for task in heldout
        if task_summary[task]["brace_adaptation"] < float(inference["severe_adaptation_regression"])
    ]
    margin = float(inference["adaptation_noninferiority_margin"])
    gates = {
        "H_preservation": pooled["anchor_main_preservation"]["lower"] > 0,
        "H_credit": pooled["credit_main_adaptation"]["lower"] > 0,
        "H_pareto": pooled["brace_preservation"]["lower"] > 0 and pooled["brace_adaptation"]["lower"] > margin,
        "H_generalization": (
            positive_tasks >= int(inference["minimum_positive_heldout_tasks"]) and not severe
        ),
        "artifact_provenance": bool(provenance.get("complete")),
    }
    if protocol["launch_gates"].get("require_local_dp_reproduction", False):
        gates["local_dp_reproduction"] = not any(
            task_summary[task]["local_dp_reproduction_deviation_gt_10pp"] for task in heldout
        )
    failed = [name for name, passed in gates.items() if not passed]
    return {
        "schema_version": 1,
        "stage": "brace_multitask_confirmatory_aggregate",
        "status": "passed" if not failed else "failed",
        "gates": gates,
        "failed_gates": failed,
        "heldout_tasks": heldout,
        "positive_brace_preservation_tasks": positive_tasks,
        "severe_adaptation_regression_tasks": severe,
        "pooled": pooled,
        "per_task": task_summary,
        "provenance": provenance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=BRACE_DIR / "multitask_protocol.v1.json")
    parser.add_argument("--artifact-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validation = validate_multitask_protocol(args.protocol, require_method_freeze=True)
    if not validation["passed"]:
        raise SystemExit("invalid or unfrozen multitask protocol: " + "; ".join(validation["errors"]))
    protocol_path = args.protocol.resolve()
    protocol = read_json(protocol_path)
    task_path = resolve_repo_path(protocol["task_manifest"])
    task_manifest = read_json(task_path)
    heldout = [str(task) for task in task_manifest["heldout_tasks"]]
    seeds = [int(seed) for seed in protocol["training"]["training_seeds"]]
    index_path = args.artifact_index.resolve()
    seed_manifests: dict[str, dict[str, Any]] = {}
    seed_manifest_hashes: dict[str, str] = {}
    for task in heldout:
        seed_path = BRACE_DIR / "seeds" / "multitask_v1" / f"{task}.json"
        if not seed_path.is_file():
            raise SystemExit(f"missing frozen seed manifest for aggregation: {seed_path}")
        seed_manifest = read_json(seed_path)
        if seed_manifest.get("status") != "frozen" or not sha256_sidecar_valid(seed_path):
            raise SystemExit(f"seed manifest is not frozen with a valid SHA256 sidecar: {seed_path}")
        seed_manifests[task] = seed_manifest
        seed_manifest_hashes[task] = file_sha256(seed_path)
    matrix, index = load_artifact_index(
        index_path,
        tasks=heldout,
        training_seeds=seeds,
        protocol=protocol,
        seed_manifests=seed_manifests,
    )
    expected = {
        "protocol_sha256": file_sha256(protocol_path),
        "task_manifest_sha256": file_sha256(task_path),
        "method_freeze_sha256": validation["method_freeze_sha256"],
    }
    mismatches = [key for key, value in expected.items() if index.get(key) != value]
    if mismatches:
        raise SystemExit(f"artifact index provenance mismatch: {mismatches}")
    provenance = {
        "complete": True,
        "artifact_index_path": str(index_path),
        "artifact_index_sha256": file_sha256(index_path),
        "seed_manifest_sha256": seed_manifest_hashes,
        **expected,
    }
    summary = aggregate_multitask(protocol, task_manifest, matrix, provenance=provenance)
    write_json_atomic(args.output, summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
