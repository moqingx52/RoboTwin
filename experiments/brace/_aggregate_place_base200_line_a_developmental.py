#!/usr/bin/env python3
"""Developmental Line A summary for Base200 place_container_plate.

Primary preservation metric is untouched_preservation under all-repeats-success
(3 repeats). Anchor contrast per training seed is:

    mean(B2, B3) - mean(N1, B1)

on the preservation rate (1 - forgetting_rate). B1-vs-N1 is reported only as a
verification diagnostic. Empty boundary/anchor_probe cohorts are recorded as
design limits, not as passed gates.

Does not freeze the method. Writes developmental_summary.json and stops there.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
BRACE = REPO / "experiments" / "brace"
DP = REPO / "policy" / "DP"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.brace.aggregate_confirmatory_preservation import forgetting_rate
from experiments.brace.replay_audit import read_json, write_json_atomic

METHODS = ("U0", "N1", "B1", "B2", "B3")
SEEDS = (1, 2, 3, 4, 5)
CHECKPOINT_LABEL = "place_base200_v2"
EPOCHS = 10
SCREEN_PROTOCOL = BRACE / "screen_protocol.v1.2.json"
PRESERVATION_PROTOCOL = BRACE / "screen_protocol.v1.2.base200_line_a_preservation.json"
MULTITASK_V2 = BRACE / "multitask_protocol.v2.json"
PRIMARY_SPLIT = "untouched_preservation"
REPEATS_REQUIRED = 3
POLICY_SEED_OFFSET = 4000


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_eval(root: Path, variant: str) -> dict[str, Any] | None:
    path = root / "place_container_plate" / f"{variant}.json"
    if not path.is_file():
        return None
    return read_json(path)


def variant_complete(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    return bool((payload.get("progress") or {}).get("complete"))


def mean_sr_from_rows(payload: dict[str, Any] | None) -> float | None:
    if payload is None:
        return None
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        splits = payload.get("splits") or {}
        for stats in splits.values():
            if isinstance(stats, dict) and stats.get("mean_sr") is not None:
                return float(stats["mean_sr"])
        return None
    oks = [1.0 if row.get("success") else 0.0 for row in rows]
    return sum(oks) / len(oks)


def adaptation_matrix(eval_root: Path) -> dict[str, Any]:
    cells = []
    missing = []
    for method in METHODS:
        for seed in SEEDS:
            for split in ("confirm_easy", "confirm_hard"):
                variant = f"line_a_{method}_seed{seed}_{split}"
                payload = load_eval(eval_root, variant)
                done = variant_complete(payload)
                cell = {
                    "method": method,
                    "seed": seed,
                    "split": split,
                    "variant": variant,
                    "complete": done,
                    "success_rate": mean_sr_from_rows(payload) if done else None,
                    "n_rows": len(payload.get("rows") or []) if payload else 0,
                }
                cells.append(cell)
                if not done:
                    missing.append(variant)
    return {
        "n_expected": len(METHODS) * len(SEEDS) * 2,
        "n_complete": sum(1 for c in cells if c["complete"]),
        "complete": not missing,
        "missing": missing,
        "cells": cells,
    }


def check_preservation_progress(payload: dict[str, Any]) -> list[str]:
    progress = payload.get("progress") or {}
    errors = []
    if int(progress.get("extra_split_repeats") or 0) != REPEATS_REQUIRED:
        errors.append(f"extra_split_repeats={progress.get('extra_split_repeats')} expected {REPEATS_REQUIRED}")
    if int(progress.get("policy_seed_offset") or -1) != POLICY_SEED_OFFSET:
        errors.append(f"policy_seed_offset={progress.get('policy_seed_offset')} expected {POLICY_SEED_OFFSET}")
    return errors


def preservation_rate_from_payload(
    payload: dict[str, Any],
    untouched: list[int],
) -> dict[str, Any]:
    rows = payload.get("rows") or []
    forget = forgetting_rate(candidate_rows=rows, cohort_seeds=untouched, split=PRIMARY_SPLIT)
    rate = None if forget["forgetting_rate"] is None else 1.0 - float(forget["forgetting_rate"])
    return {
        "split": PRIMARY_SPLIT,
        "rule": "all_repeats_success",
        "repeats_required": REPEATS_REQUIRED,
        "preservation_rate": rate,
        "forgetting": forget,
        "progress_errors": check_preservation_progress(payload),
    }


def preservation_matrix(eval_root: Path, untouched: list[int]) -> dict[str, Any]:
    cells = []
    missing = []
    variants = [("base", 0, "line_a_base_preservation")]
    for method in METHODS:
        for seed in SEEDS:
            variants.append((method, seed, f"line_a_{method}_seed{seed}_preservation"))
    for method, seed, variant in variants:
        payload = load_eval(eval_root, variant)
        done = variant_complete(payload)
        cell: dict[str, Any] = {
            "method": method,
            "seed": seed,
            "variant": variant,
            "complete": done,
        }
        if done and payload is not None:
            try:
                cell.update(preservation_rate_from_payload(payload, untouched))
            except ValueError as exc:
                cell["error"] = str(exc)
                missing.append(variant)
        else:
            missing.append(variant)
        cells.append(cell)
    return {
        "n_expected": 1 + len(METHODS) * len(SEEDS),
        "n_complete": sum(1 for c in cells if c["complete"] and not c.get("error")),
        "complete": not missing,
        "missing": missing,
        "primary_split": PRIMARY_SPLIT,
        "cells": cells,
    }


def rate_for(cells: list[dict[str, Any]], method: str, seed: int) -> float | None:
    for cell in cells:
        if cell.get("method") == method and int(cell.get("seed", -1)) == int(seed):
            return cell.get("preservation_rate")
    return None


def behavioral_anchor_gate(pres: dict[str, Any]) -> dict[str, Any]:
    cells = pres.get("cells") or []
    pairs = []
    wins = 0
    if not pres.get("complete"):
        return {
            "passed": False,
            "complete": False,
            "paired_seed_count": 0,
            "directional_wins": 0,
            "minimum_directional_wins": 4,
            "effect_point_estimate": None,
            "contrast": "mean(B2,B3)-mean(N1,B1) on untouched preservation_rate",
            "reason": "preservation_eval_incomplete",
            "pairs": pairs,
        }
    for seed in SEEDS:
        n1 = rate_for(cells, "N1", seed)
        b1 = rate_for(cells, "B1", seed)
        b2 = rate_for(cells, "B2", seed)
        b3 = rate_for(cells, "B3", seed)
        if None in (n1, b1, b2, b3):
            return {
                "passed": False,
                "complete": False,
                "paired_seed_count": 0,
                "directional_wins": 0,
                "minimum_directional_wins": 4,
                "effect_point_estimate": None,
                "contrast": "mean(B2,B3)-mean(N1,B1) on untouched preservation_rate",
                "reason": f"missing_arm_rate_seed{seed}",
                "pairs": pairs,
            }
        no_anchor = (float(n1) + float(b1)) / 2.0
        anchor = (float(b2) + float(b3)) / 2.0
        delta = anchor - no_anchor
        win = delta > 0
        if win:
            wins += 1
        pairs.append(
            {
                "seed": seed,
                "N1": n1,
                "B1": b1,
                "B2": b2,
                "B3": b3,
                "mean_no_anchor": no_anchor,
                "mean_anchor": anchor,
                "delta": delta,
                "win": win,
            }
        )
    deltas = [p["delta"] for p in pairs]
    effect = sum(deltas) / len(deltas)
    verification = []
    for seed in SEEDS:
        b1 = rate_for(cells, "B1", seed)
        n1 = rate_for(cells, "N1", seed)
        verification.append(
            {
                "seed": seed,
                "B1": b1,
                "N1": n1,
                "delta": None if b1 is None or n1 is None else float(b1) - float(n1),
            }
        )
    return {
        "passed": wins >= 4 and effect > 0,
        "complete": True,
        "paired_seed_count": 5,
        "directional_wins": wins,
        "minimum_directional_wins": 4,
        "effect_point_estimate": effect,
        "contrast": "mean(B2,B3)-mean(N1,B1) on untouched preservation_rate",
        "pairs": pairs,
        "verification_B1_minus_N1": verification,
        "verification_note": "B1-N1 measures verification, not the anchor preservation gate.",
    }


def joint_gate(
    *,
    feasibility_inv: dict[str, Any] | None,
    screen: dict[str, Any],
    behavioral: dict[str, Any],
    adaptation: dict[str, Any],
    preservation: dict[str, Any],
    cohort_limits: dict[str, Any],
) -> dict[str, Any]:
    feas_required = bool(
        (screen.get("screens") or {}).get("preservation", {}).get("requires_constraint_feasibility")
    )
    ineligible_if_infeasible = bool(
        (screen.get("constraint_feasibility") or {}).get("ineligible_if_infeasible")
    )
    feas_passed = False
    feas_summary = None
    if feasibility_inv is not None:
        feas_summary = feasibility_inv.get("summary") or {}
        feas_passed = int(feas_summary.get("n_failed", 1)) == 0
    reasons = []
    if not adaptation.get("complete"):
        reasons.append("adaptation_eval_incomplete")
    if not preservation.get("complete"):
        reasons.append("preservation_eval_incomplete")
    if feas_required and ineligible_if_infeasible and not feas_passed:
        reasons.append("constraint_feasibility_failed")
    if not behavioral.get("passed"):
        reasons.append("behavioral_preservation_gate_failed")
    complete = bool(
        adaptation.get("complete") and preservation.get("complete") and feasibility_inv is not None
    )
    scientific_go = complete and not reasons
    freeze_reasons = list(reasons) + ["stop_line_no_method_freeze"]
    return {
        "passed": scientific_go,
        "complete": complete,
        "reasons": reasons,
        "constraint_feasibility_required": feas_required,
        "ineligible_if_infeasible": ineligible_if_infeasible,
        "constraint_feasibility_passed": feas_passed,
        "behavioral_preservation_passed": bool(behavioral.get("passed")),
        "empty_secondary_splits": cohort_limits.get("empty_secondary_splits") or [],
        "feasibility_summary": feas_summary,
        "line_a_conclusion": "go" if scientific_go else "no-go",
        "freeze_blocked": True,
        "freeze_blocked_until": freeze_reasons,
        "protocol_note": (
            "Under frozen screen.v1.2, feasibility failures make B2/B3 gate-ineligible. "
            "Empty boundary/anchor_probe are design limits, not passed gates. "
            "This developmental summary does not freeze or promote the method."
        ),
    }


def load_jsonl_stats(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    env_seeds = sorted({int(r["env_seed"]) for r in rows if "env_seed" in r})
    point_types: dict[str, int] = {}
    for row in rows:
        key = str(row.get("point_type") or row.get("type") or "unknown")
        point_types[key] = point_types.get(key, 0) + 1
    return {
        "path": str(path),
        "exists": True,
        "n_chunks": len(rows),
        "n_env_seeds": len(env_seeds),
        "env_seeds": env_seeds,
        "point_types": point_types,
        "sha256": file_sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.run_dir is None:
        ptr = BRACE / "runs/LATEST_place_base200_v2_line_a_pilot"
        run_dir = Path(ptr.read_text(encoding="utf-8").strip())
        if not run_dir.is_absolute():
            run_dir = REPO / run_dir
    else:
        run_dir = args.run_dir if args.run_dir.is_absolute() else REPO / args.run_dir

    screen = read_json(SCREEN_PROTOCOL)
    pres_protocol = read_json(PRESERVATION_PROTOCOL) if PRESERVATION_PROTOCOL.is_file() else {}
    multitask_v2 = read_json(MULTITASK_V2) if MULTITASK_V2.is_file() else {}

    feas_inv_path = run_dir / "constraint_feasibility_inventory.json"
    feasibility_inv = read_json(feas_inv_path) if feas_inv_path.is_file() else None

    cohort_ptr = BRACE / "runs/LATEST_place_base200_line_a_preservation_cohort"
    cohort = None
    cohort_path = None
    if cohort_ptr.is_file():
        cohort_path = Path(cohort_ptr.read_text(encoding="utf-8").strip())
        if not cohort_path.is_absolute():
            cohort_path = REPO / cohort_path
        if cohort_path.is_file():
            cohort = read_json(cohort_path)
    cohorts = (cohort or {}).get("cohorts") or {}
    untouched = [int(s) for s in cohorts.get("untouched_preservation") or []]
    sizes = {k: len(v) if isinstance(v, list) else v for k, v in cohorts.items()}
    empty_secondary = [k for k in ("boundary", "anchor_probe") if not cohorts.get(k)]
    cohort_limits = {
        "partition_sizes": sizes,
        "empty_secondary_splits": empty_secondary,
        "meets_min_untouched": bool((cohort or {}).get("meets_min_untouched")),
        "frozen": bool((cohort or {}).get("frozen")),
        "note": (
            "Empty boundary and anchor_probe do not block the untouched primary endpoint, "
            "but those secondary gates must not be treated as passed."
        ),
    }

    adaptation = adaptation_matrix(run_dir / "eval")
    preservation = preservation_matrix(run_dir / "eval_preservation", untouched)
    behavioral = behavioral_anchor_gate(preservation)
    gate = joint_gate(
        feasibility_inv=feasibility_inv,
        screen=screen,
        behavioral=behavioral,
        adaptation=adaptation,
        preservation=preservation,
        cohort_limits=cohort_limits,
    )
    summary = {
        "schema_version": 2,
        "kind": "place_base200_line_a_developmental_summary",
        "task": "place_container_plate",
        "stage": "brace_v2_intervention_pilot",
        "substrate": "Base200",
        "status": "passed" if gate["line_a_conclusion"] == "go" else "failed",
        "complete": bool(gate.get("complete")),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dir": str(run_dir),
        "protocol_revision_screen": screen.get("protocol_revision"),
        "protocol_revision_preservation": pres_protocol.get("protocol_revision"),
        "protocol_revision_multitask": multitask_v2.get("protocol_revision"),
        "multitask_protocol_status": multitask_v2.get("status"),
        "stop_line": {
            "gpu_pipeline_stopped_after": "developmental_summary",
            "method_freeze": "not_run",
            "heldout_brace": "not_run",
            "track4_probes": "not_run",
        },
        "adaptation_eval": {
            "state": str(run_dir / "eval_state.json"),
            "output_dir": str(run_dir / "eval"),
            **{k: adaptation[k] for k in ("n_expected", "n_complete", "complete", "missing")},
            "cells": adaptation["cells"],
        },
        "preservation_eval": {
            "state": str(run_dir / "preservation_eval_state.json"),
            "output_dir": str(run_dir / "eval_preservation"),
            "logs_dir": str(run_dir / "logs_preservation"),
            "cohort_path": str(cohort_path) if cohort_path else None,
            "cohort_sha256": file_sha256(cohort_path) if cohort_path else None,
            "policy_seed_offset": POLICY_SEED_OFFSET,
            "extra_split_repeats": REPEATS_REQUIRED,
            **{k: preservation[k] for k in ("n_expected", "n_complete", "complete", "missing", "primary_split")},
            "cells": preservation["cells"],
        },
        "cohort_limits": cohort_limits,
        "constraint_feasibility_inventory_path": str(feas_inv_path) if feas_inv_path.is_file() else None,
        "constraint_feasibility": feasibility_inv.get("summary") if feasibility_inv else None,
        "preservation_go_no_go": behavioral,
        "joint_line_a_gate": gate,
        "method_budget": {
            "B1": load_jsonl_stats(BRACE / "datasets/place_base200_v2_B1.jsonl"),
            "N1": load_jsonl_stats(BRACE / "datasets/place_base200_v2_N1.jsonl"),
        },
    }
    out = run_dir / "developmental_summary.json"
    write_json_atomic(out, summary)
    pointer = BRACE / "runs/LATEST_place_base200_line_a_developmental_summary"
    pointer.write_text(str(out) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(out),
                "status": summary["status"],
                "complete": summary["complete"],
                "line_a_conclusion": gate["line_a_conclusion"],
                "freeze_blocked": gate["freeze_blocked"],
                "adaptation_complete": adaptation["complete"],
                "preservation_complete": preservation["complete"],
                "feasibility_n_failed": (feasibility_inv or {}).get("summary", {}).get("n_failed"),
                "behavioral": {
                    "passed": behavioral.get("passed"),
                    "wins": behavioral.get("directional_wins"),
                    "effect": behavioral.get("effect_point_estimate"),
                },
            },
            indent=2,
        )
    )
    if not summary["complete"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
