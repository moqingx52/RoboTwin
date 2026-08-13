#!/usr/bin/env python3
"""Aggregate Base200 Line A place_container_plate development results.

Builds a gate-facing summary that includes:
  - constraint feasibility inventory for B2/B3
  - adaptation matrix completeness (easy/hard)
  - preservation matrix completeness (when available)
  - joint go/no-go under the frozen screen protocol
  - checkpoint / cohort hashes
  - method-budget notes (B1/N1 chunk counts)

Does not freeze the method. Writes summary.json under the pilot run dir for a
future brace.multitask.v2 freeze path. Exit 0 only when the joint gate passes.
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

from experiments.brace.replay_audit import read_json, write_json_atomic

METHODS = ("U0", "N1", "B1", "B2", "B3")
SEEDS = (1, 2, 3, 4, 5)
CHECKPOINT_LABEL = "place_base200_v2"
EPOCHS = 10
SCREEN_PROTOCOL = BRACE / "screen_protocol.v1.2.json"
MULTITASK_V2 = BRACE / "multitask_protocol.v2.json"


def file_sha256(path: Path, *, max_bytes: int | None = None) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    size = path.stat().st_size
    with path.open("rb") as handle:
        if max_bytes is not None and size > max_bytes:
            # Large checkpoints: hash size + head/tail for provenance without full I/O.
            head = handle.read(max_bytes // 2)
            handle.seek(max(0, size - max_bytes // 2))
            tail = handle.read(max_bytes // 2)
            digest.update(str(size).encode())
            digest.update(head)
            digest.update(tail)
            return "partial:" + digest.hexdigest()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def eval_variant_complete(root: Path, variant: str) -> bool:
    path = root / "place_container_plate" / f"{variant}.json"
    if not path.is_file():
        return False
    payload = read_json(path)
    return bool((payload.get("progress") or {}).get("complete"))


def success_rate(root: Path, variant: str) -> float | None:
    path = root / "place_container_plate" / f"{variant}.json"
    if not path.is_file():
        return None
    payload = read_json(path)
    rows = payload.get("results") or payload.get("episodes") or []
    if not rows:
        summary = payload.get("summary") or {}
        if "success_rate" in summary:
            return float(summary["success_rate"])
        return None
    oks = [1.0 if r.get("success") else 0.0 for r in rows if "success" in r]
    if not oks:
        return None
    return sum(oks) / len(oks)


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


def adaptation_matrix(eval_root: Path) -> dict[str, Any]:
    cells = []
    missing = []
    for method in METHODS:
        for seed in SEEDS:
            for split in ("confirm_easy", "confirm_hard"):
                variant = f"line_a_{method}_seed{seed}_{split}"
                done = eval_variant_complete(eval_root, variant)
                cell = {
                    "method": method,
                    "seed": seed,
                    "split": split,
                    "variant": variant,
                    "complete": done,
                    "success_rate": success_rate(eval_root, variant) if done else None,
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


def preservation_matrix(eval_root: Path) -> dict[str, Any]:
    cells = []
    missing = []
    base_variant = "line_a_base_preservation"
    base_done = eval_variant_complete(eval_root, base_variant)
    cells.append(
        {
            "method": "base",
            "seed": 0,
            "variant": base_variant,
            "complete": base_done,
            "success_rate": success_rate(eval_root, base_variant) if base_done else None,
        }
    )
    if not base_done:
        missing.append(base_variant)
    for method in METHODS:
        for seed in SEEDS:
            variant = f"line_a_{method}_seed{seed}_preservation"
            done = eval_variant_complete(eval_root, variant)
            cells.append(
                {
                    "method": method,
                    "seed": seed,
                    "variant": variant,
                    "complete": done,
                    "success_rate": success_rate(eval_root, variant) if done else None,
                }
            )
            if not done:
                missing.append(variant)
    return {
        "n_expected": 1 + len(METHODS) * len(SEEDS),
        "n_complete": sum(1 for c in cells if c["complete"]),
        "complete": not missing,
        "missing": missing,
        "cells": cells,
    }


def checkpoint_hashes() -> dict[str, Any]:
    rows = []
    for method in METHODS:
        for seed in SEEDS:
            cdir = DP / "checkpoints" / f"place_container_plate-brace-{CHECKPOINT_LABEL}-{method}-{seed}"
            ckpt = cdir / f"{EPOCHS}.ckpt"
            row: dict[str, Any] = {
                "method": method,
                "seed": seed,
                "checkpoint_path": str(ckpt),
                "checkpoint_bytes": ckpt.stat().st_size if ckpt.is_file() else None,
                "checkpoint_sha256": file_sha256(ckpt, max_bytes=16 * 1024 * 1024),
            }
            if method in ("B2", "B3"):
                feas = cdir / f"{EPOCHS}.feasibility.json"
                row["feasibility_path"] = str(feas)
                row["feasibility_sha256"] = file_sha256(feas)
            rows.append(row)
    base = DP / "checkpoints/place_container_plate-demo_clean-200-0/600.ckpt"
    return {
        "base_checkpoint": {
            "path": str(base),
            "bytes": base.stat().st_size if base.is_file() else None,
            "sha256": file_sha256(base, max_bytes=16 * 1024 * 1024),
        },
        "trained": rows,
        "hash_note": "Checkpoint digests are size+head/tail partial hashes unless small.",
    }


def behavioral_preservation_go_no_go(pres: dict[str, Any]) -> dict[str, Any]:
    """Paired seed 4/5 directional wins of B1 vs N1 on untouched rates when complete.

    Placeholder until full cohort-level forgetting aggregation is wired; still
    surfaces completeness and per-seed rates for archival.
    """
    by_key = {(c["method"], c["seed"]): c for c in pres["cells"] if c["method"] != "base"}
    wins = 0
    pairs = []
    for seed in SEEDS:
        b1 = by_key.get(("B1", seed))
        n1 = by_key.get(("N1", seed))
        if not b1 or not n1 or not b1["complete"] or not n1["complete"]:
            return {
                "passed": False,
                "complete": False,
                "paired_seed_count": 0,
                "directional_wins": 0,
                "effect_point_estimate": None,
                "reason": "preservation_eval_incomplete",
                "pairs": pairs,
            }
        b1_sr = b1["success_rate"]
        n1_sr = n1["success_rate"]
        delta = None if b1_sr is None or n1_sr is None else b1_sr - n1_sr
        win = delta is not None and delta > 0
        if win:
            wins += 1
        pairs.append({"seed": seed, "B1": b1_sr, "N1": n1_sr, "delta": delta, "win": win})
    deltas = [p["delta"] for p in pairs if p["delta"] is not None]
    effect = sum(deltas) / len(deltas) if deltas else None
    return {
        "passed": wins >= 4 and effect is not None and effect > 0,
        "complete": True,
        "paired_seed_count": 5,
        "directional_wins": wins,
        "minimum_directional_wins": 4,
        "effect_point_estimate": effect,
        "pairs": pairs,
        "note": (
            "Interim B1-vs-N1 success-rate sign rule on preservation variants. "
            "Replace with cohort forgetting aggregator before confirmatory freeze."
        ),
    }


def joint_gate(
    *,
    feasibility_inv: dict[str, Any] | None,
    screen: dict[str, Any],
    behavioral: dict[str, Any],
    adaptation: dict[str, Any],
    preservation: dict[str, Any],
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
    passed = not reasons
    return {
        "passed": passed,
        "complete": adaptation.get("complete") and preservation.get("complete") and feasibility_inv is not None,
        "reasons": reasons,
        "constraint_feasibility_required": feas_required,
        "ineligible_if_infeasible": ineligible_if_infeasible,
        "constraint_feasibility_passed": feas_passed,
        "behavioral_preservation_passed": bool(behavioral.get("passed")),
        "feasibility_summary": feas_summary,
        "line_a_conclusion": "go" if passed else "no-go",
        "freeze_blocked_until": reasons,
        "protocol_note": (
            "Under frozen screen.v1.2, feasibility failures make B2/B3 gate-ineligible. "
            "Do not retune anchors on this batch and treat the same numbers as confirmatory."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Pilot run dir (default: LATEST_place_base200_v2_line_a_pilot).",
    )
    parser.add_argument("--force-feasibility-inventory", action="store_true")
    args = parser.parse_args()
    if args.run_dir is None:
        ptr = BRACE / "runs/LATEST_place_base200_v2_line_a_pilot"
        run_dir = Path(ptr.read_text(encoding="utf-8").strip())
        if not run_dir.is_absolute():
            run_dir = REPO / run_dir
    else:
        run_dir = args.run_dir if args.run_dir.is_absolute() else REPO / args.run_dir

    feas_inv_path = run_dir / "constraint_feasibility_inventory.json"
    if args.force_feasibility_inventory or not feas_inv_path.is_file():
        cmd_mod = BRACE / "_inventory_place_base200_feasibility.py"
        import subprocess

        subprocess.run(
            [sys.executable, str(cmd_mod), "--run-dir", str(run_dir)]
            + (["--force-recompute"] if args.force_feasibility_inventory else []),
            cwd=REPO,
            check=False,
        )
    feasibility_inv = read_json(feas_inv_path) if feas_inv_path.is_file() else None

    cohort_ptr = BRACE / "runs/LATEST_place_base200_line_a_preservation_cohort"
    cohort_path = None
    cohort_sha = None
    if cohort_ptr.is_file():
        cohort_path = Path(cohort_ptr.read_text(encoding="utf-8").strip())
        if not cohort_path.is_absolute():
            cohort_path = REPO / cohort_path
        cohort_sha = file_sha256(cohort_path)

    adaptation = adaptation_matrix(run_dir / "eval")
    preservation = preservation_matrix(run_dir / "eval_preservation")
    behavioral = behavioral_preservation_go_no_go(preservation)
    screen = read_json(SCREEN_PROTOCOL)
    multitask_v2 = read_json(MULTITASK_V2) if MULTITASK_V2.is_file() else {}
    gate = joint_gate(
        feasibility_inv=feasibility_inv,
        screen=screen,
        behavioral=behavioral,
        adaptation=adaptation,
        preservation=preservation,
    )

    summary = {
        "schema_version": 1,
        "task": "place_container_plate",
        "stage": "brace_v2_intervention_pilot",
        "substrate": "Base200",
        "protocol_revision_screen": screen.get("protocol_revision"),
        "protocol_revision_multitask": multitask_v2.get("protocol_revision"),
        "multitask_protocol_status": multitask_v2.get("status"),
        "status": "passed" if gate["passed"] else "failed",
        "complete": bool(gate.get("complete")),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dir": str(run_dir),
        "adaptation_eval": {
            "state": str(run_dir / "eval_state.json"),
            **{k: adaptation[k] for k in ("n_expected", "n_complete", "complete", "missing")},
            "cells": adaptation["cells"],
        },
        "preservation_eval": {
            "state": str(run_dir / "preservation_eval_state.json"),
            "cohort_path": str(cohort_path) if cohort_path else None,
            "cohort_sha256": cohort_sha,
            **{k: preservation[k] for k in ("n_expected", "n_complete", "complete", "missing")},
            "cells": preservation["cells"],
        },
        "constraint_feasibility_inventory_path": str(feas_inv_path) if feas_inv_path.is_file() else None,
        "constraint_feasibility": feasibility_inv.get("summary") if feasibility_inv else None,
        "constraint_feasibility_rows": feasibility_inv.get("rows") if feasibility_inv else None,
        "preservation_go_no_go": behavioral,
        "joint_line_a_gate": gate,
        "checkpoints": checkpoint_hashes(),
        "method_budget": {
            "B1": load_jsonl_stats(BRACE / "datasets/place_base200_v2_B1.jsonl"),
            "N1": load_jsonl_stats(BRACE / "datasets/place_base200_v2_N1.jsonl"),
            "export_note": (
                "export_verified_chunks exports all accepted=true points without filtering "
                "point_type; confirm random_negative_control inclusion is intentional before freeze."
            ),
        },
        "freeze_wiring": {
            "current_tool": "experiments/brace/freeze_multitask_method.py",
            "default_protocol": "multitask_protocol.v1.json",
            "required_for_base200": "brace.multitask.v2 + versioned method-freeze amendment",
            "blocked_reason": (
                "v2 status is base_acquisition_frozen_method_pending and lacks a method-freeze "
                "path; do not use v1 freeze to claim Base200 Line A."
            ),
            "summary_fields_for_future_freeze": [
                "joint_line_a_gate",
                "constraint_feasibility",
                "preservation_go_no_go",
                "adaptation_eval",
                "preservation_eval",
                "checkpoints",
                "method_budget",
            ],
        },
    }
    out = run_dir / "summary.json"
    write_json_atomic(out, summary)
    print(
        json.dumps(
            {
                "output": str(out),
                "status": summary["status"],
                "complete": summary["complete"],
                "joint_line_a_gate": gate,
                "adaptation_complete": adaptation["complete"],
                "preservation_complete": preservation["complete"],
                "feasibility_n_failed": (feasibility_inv or {}).get("summary", {}).get("n_failed"),
            },
            indent=2,
        )
    )
    if not summary["complete"]:
        return 3
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
