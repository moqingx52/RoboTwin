"""Shared BRACE developmental screen gate helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_ANCHOR_GROUPS = ("base_solved", "boundary")

# NOTE: screen.v1.2 reuses anchor_smoke.identity_epsilon for feasibility gating.
# That value validates implementation correctness (same-weight student≈teacher),
# not a behavior-calibrated preservation budget. See Phase 2 protocol split.


def hard_for_checkpoint_selection(protocol: dict[str, Any]) -> bool:
    return bool(protocol.get("eval", {}).get("hard_for_checkpoint_selection", True))


def selection_splits(protocol: dict[str, Any]) -> tuple[str, ...]:
    if hard_for_checkpoint_selection(protocol):
        return ("id_heldout", "train_seen", "hard_20")
    return ("id_heldout", "train_seen")


def selection_score(metrics: dict[str, float], base: dict[str, float], protocol: dict[str, Any]) -> float:
    splits = selection_splits(protocol)
    normalized = [metrics[split] / max(base[split], 0.05 if split == "hard_20" else base[split]) for split in splits]
    return float(min(normalized))


def parse_training_logs(log_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not log_path.is_file():
        return rows
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def summarize_feasibility_trajectory(
    rows: list[dict[str, Any]],
    *,
    epsilon: float,
    tail_fraction: float = 0.2,
    tail_steps: int | None = None,
    prefix: str = "",
) -> dict[str, Any]:
    """Forensic window stats; not used for frozen screen.v1.2 pass/fail."""
    if not rows:
        return {"rows": 0}

    if tail_steps is not None:
        tail_count = max(1, min(int(tail_steps), len(rows)))
        tail_mode = "steps"
    else:
        tail_count = max(1, int(math.ceil(len(rows) * tail_fraction)))
        tail_mode = "fraction"
    warmup_rows = rows[:-tail_count] if len(rows) > tail_count else []
    tail_rows = rows[-tail_count:]
    summary: dict[str, Any] = {
        "rows": len(rows),
        "tail_mode": tail_mode,
        "tail_rows": len(tail_rows),
    }
    if tail_mode == "steps":
        summary["tail_steps"] = tail_count
    else:
        summary["tail_fraction"] = tail_fraction

    for group in REQUIRED_ANCHOR_GROUPS:
        train_key = f"brace_constraint/{group}"
        warmup_vals = [float(row[train_key]) for row in warmup_rows if train_key in row]
        tail_vals = [float(row[train_key]) for row in tail_rows if train_key in row]
        all_vals = [float(row[train_key]) for row in rows if train_key in row]
        ema_key = f"brace_monitor/{group}_ema_drift"
        tail_ema = [float(row[ema_key]) for row in tail_rows if ema_key in row]
        group_summary: dict[str, Any] = {}
        if warmup_vals:
            group_summary[f"{prefix}warmup_max"] = float(max(warmup_vals))
        if tail_vals:
            arr = np.asarray(tail_vals, dtype=np.float64)
            group_summary[f"{prefix}tail_mean"] = float(arr.mean())
            group_summary[f"{prefix}tail_p90"] = float(np.percentile(arr, 90))
            group_summary[f"{prefix}tail_violation_fraction"] = float(np.mean(arr > epsilon))
            if len(tail_vals) >= 2:
                xs = np.arange(len(tail_vals), dtype=np.float64)
                slope = float(np.polyfit(xs, arr, 1)[0])
                group_summary[f"{prefix}tail_slope"] = slope
        if all_vals:
            group_summary[f"{prefix}full_mean"] = float(np.mean(all_vals))
            group_summary[f"{prefix}full_p90"] = float(np.percentile(all_vals, 90))
        if tail_ema:
            group_summary[f"{prefix}tail_ema_mean"] = float(np.mean(tail_ema))
        dual_key = f"brace_dual/{group}"
        if rows and dual_key in rows[-1]:
            group_summary[f"{prefix}dual_final"] = float(rows[-1][dual_key])
        summary[group] = group_summary

    dual_keys = [key for key in rows[-1] if key.startswith("brace_dual/")]
    if dual_keys:
        summary[f"{prefix}dual_final"] = {key.split("/", 1)[1]: float(rows[-1][key]) for key in dual_keys}
    return summary


def evaluate_constraint_feasibility(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    *,
    required_groups: tuple[str, ...] = REQUIRED_ANCHOR_GROUPS,
) -> dict[str, Any]:
    gate = protocol.get("constraint_feasibility", {})
    epsilon = float(protocol.get("anchor_smoke", {}).get("identity_epsilon", 1e-4))
    mean_factor = float(gate.get("mean_tolerance_factor", 2.0))
    p90_factor = float(gate.get("p90_max_factor", 5.0))
    violation_max = gate.get("violation_fraction_max", {})
    require_raw_and_ema = bool(gate.get("require_raw_and_ema", False))

    groups: dict[str, Any] = {}
    missing_groups = []
    for group in required_groups:
        raw_values = [
            float(row[f"brace_constraint/{group}"])
            for row in rows
            if f"brace_constraint/{group}" in row
        ]
        if not raw_values:
            missing_groups.append(group)
            continue
        arr = np.asarray(raw_values, dtype=np.float64)
        ema_values = [
            float(row[f"brace_monitor/{group}_ema_drift"])
            for row in rows
            if f"brace_monitor/{group}_ema_drift" in row
        ]
        ema_arr = np.asarray(ema_values, dtype=np.float64) if ema_values else None
        group_result = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "violation_fraction": float(np.mean(arr > epsilon)),
            "epsilon": epsilon,
            "mean_pass": float(arr.mean()) <= epsilon * mean_factor,
            "p90_pass": float(np.percentile(arr, 90)) <= epsilon * p90_factor,
            "violation_fraction_pass": float(np.mean(arr > epsilon))
            <= float(violation_max.get(group, 0.5)),
        }
        if require_raw_and_ema:
            if ema_arr is None or ema_arr.size == 0:
                group_result["ema_pass"] = False
                group_result["ema_missing"] = True
            else:
                group_result["ema_mean"] = float(ema_arr.mean())
                group_result["ema_p90"] = float(np.percentile(ema_arr, 90))
                group_result["ema_pass"] = float(ema_arr.mean()) <= epsilon * mean_factor
                group_result["ema_missing"] = False
            group_result["raw_pass"] = group_result["mean_pass"] and group_result["p90_pass"]
            group_result["passed"] = (
                group_result["raw_pass"]
                and group_result["violation_fraction_pass"]
                and group_result.get("ema_pass", True)
            )
        else:
            group_result["passed"] = (
                group_result["mean_pass"]
                and group_result["p90_pass"]
                and group_result["violation_fraction_pass"]
            )
        groups[group] = group_result

    passed = not missing_groups and all(group["passed"] for group in groups.values())
    return {
        "passed": passed,
        "missing_groups": missing_groups,
        "groups": groups,
        "require_raw_and_ema": require_raw_and_ema,
    }


def feasibility_artifact_path(ckpt_path: Path) -> Path:
    return ckpt_path.with_suffix(".feasibility.json")


def load_or_evaluate_feasibility(
    ckpt_path: Path,
    protocol: dict[str, Any],
    *,
    training_log: Path | None = None,
) -> dict[str, Any]:
    artifact = feasibility_artifact_path(ckpt_path)
    if artifact.is_file():
        return json.loads(artifact.read_text(encoding="utf-8"))
    if training_log is None:
        return {
            "passed": False,
            "missing_groups": list(REQUIRED_ANCHOR_GROUPS),
            "groups": {},
            "error": "missing_feasibility_artifact",
        }
    rows = parse_training_logs(training_log)
    summary = evaluate_constraint_feasibility(rows, protocol)
    artifact.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def episode_outcomes(eval_path: Path, split: str) -> dict[int, bool]:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    by_seed: dict[int, list[bool]] = {}
    for row in payload.get("rows", []):
        if row.get("split") != split:
            continue
        by_seed.setdefault(int(row["env_seed"]), []).append(
            bool(row.get("success", row.get("task_success", False)))
        )
    # Preservation is seed-level and conservative: a seed remains solved only
    # when every preregistered stochastic repeat succeeds.
    return {seed: all(outcomes) for seed, outcomes in by_seed.items()}


def resolve_eval_split(protocol: dict[str, Any], *eval_paths: Path) -> str:
    preservation_cfg = protocol.get("preservation_metrics", {})
    preferred = preservation_cfg.get("preferred_split", "base_solved_anchor")
    fallback = preservation_cfg.get("eval_split", "train_seen")
    for split in (preferred, fallback):
        if any(episode_outcomes(path, split) for path in eval_paths):
            return split
    return fallback


def compute_forgetting_gate(
    *,
    base_eval: Path,
    u1_eval: Path,
    b2_eval: Path,
    base_solved_seeds: set[int],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    preservation_cfg = protocol.get("preservation_metrics", {})
    min_paired_seeds = int(preservation_cfg.get("min_paired_seeds", 5))
    require_b2_below_u1 = bool(preservation_cfg.get("require_b2_forgetting_below_u1", True))
    split = resolve_eval_split(protocol, base_eval, u1_eval, b2_eval)

    if not base_solved_seeds:
        return {
            "passed": False,
            "reason": "missing_base_solved_seed_manifest",
            "forgetting_rate_u1": None,
            "forgetting_rate_b2": None,
            "paired_seeds": 0,
            "min_paired_seeds": min_paired_seeds,
            "split": split,
        }

    base = episode_outcomes(base_eval, split)
    u1 = episode_outcomes(u1_eval, split)
    b2 = episode_outcomes(b2_eval, split)
    seeds = sorted(base_solved_seeds & set(base) & set(u1) & set(b2))
    if len(seeds) < min_paired_seeds:
        return {
            "passed": False,
            "reason": "insufficient_paired_seeds",
            "forgetting_rate_u1": None,
            "forgetting_rate_b2": None,
            "paired_seeds": len(seeds),
            "min_paired_seeds": min_paired_seeds,
            "split": split,
        }

    forgetting_u1 = 0
    forgetting_b2 = 0
    newly_solved_u1 = 0
    newly_solved_b2 = 0
    for seed in seeds:
        if not base[seed]:
            continue
        if not u1[seed]:
            forgetting_u1 += 1
        if not b2[seed]:
            forgetting_b2 += 1
        if u1[seed] and not base[seed]:
            newly_solved_u1 += 1
        if b2[seed] and not base[seed]:
            newly_solved_b2 += 1

    base_success_count = sum(1 for seed in seeds if base[seed])
    if base_success_count == 0:
        return {
            "passed": False,
            "reason": "no_base_success_on_paired_seeds",
            "forgetting_rate_u1": None,
            "forgetting_rate_b2": None,
            "paired_seeds": len(seeds),
            "min_paired_seeds": min_paired_seeds,
            "split": split,
        }

    forgetting_rate_u1 = forgetting_u1 / base_success_count
    forgetting_rate_b2 = forgetting_b2 / base_success_count
    if require_b2_below_u1:
        passed = forgetting_rate_b2 < forgetting_rate_u1 or (
            forgetting_rate_b2 == 0.0 and forgetting_rate_u1 == 0.0
        )
        failure_reason = None if passed else "b2_forgetting_not_below_u1"
    else:
        max_delta = float(preservation_cfg.get("base_solved_forgetting_max_delta", 0.05))
        passed = forgetting_rate_b2 <= max_delta
        failure_reason = None if passed else "b2_forgetting_above_max_delta"

    return {
        "passed": passed,
        "failure_reason": failure_reason,
        "forgetting_rate_u1": forgetting_rate_u1,
        "forgetting_rate_b2": forgetting_rate_b2,
        "forgetting_count_u1": forgetting_u1,
        "forgetting_count_b2": forgetting_b2,
        "newly_solved_count_u1": newly_solved_u1,
        "newly_solved_count_b2": newly_solved_b2,
        "paired_seeds": len(seeds),
        "base_success_count": base_success_count,
        "min_paired_seeds": min_paired_seeds,
        "split": split,
        "require_b2_forgetting_below_u1": require_b2_below_u1,
    }


def load_base_solved_seeds(run_dir: Path, task: str) -> set[int]:
    manifest = run_dir / "datasets" / f"{task}_anchor_replay.zarr" / "brace_anchor_manifest.json"
    if not manifest.is_file():
        return set()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        int(row["env_seed"])
        for row in payload.get("episodes", [])
        if row.get("preservation_group") == "base_solved"
    }
