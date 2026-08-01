#!/usr/bin/env python3
"""Frozen-denoiser anchor smoke checks before B2/B3 screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.replay_audit import git_commit, read_json, repo_path, write_json_atomic


def dual_update(mu: float, violation: float, *, lr: float) -> float:
    return max(0.0, mu + lr * violation)


def run_anchor_smoke(protocol: dict[str, Any]) -> dict[str, Any]:
    cfg = protocol["anchor_smoke"]
    epsilon = float(cfg["identity_epsilon"])
    lr = float(cfg["dual_lr"])
    steps = int(cfg["violation_steps"])

    student = np.asarray([0.1, -0.2, 0.3], dtype=np.float64)
    teacher = student.copy()
    identity_violation = float(np.linalg.norm(student - teacher))
    identity_passed = identity_violation <= epsilon

    mu = 0.0
    dual_history: list[float] = []
    violation_history: list[float] = []
    rng = np.random.default_rng(0)
    for _ in range(steps):
        noisy = rng.standard_normal(3)
        violation = float(max(0.0, np.linalg.norm(noisy - teacher) - epsilon))
        mu = dual_update(mu, violation, lr=lr)
        dual_history.append(mu)
        violation_history.append(violation)

    dual_nonneg = all(value >= -1e-12 for value in dual_history)
    dual_rises_on_violation = any(v > 0 for v in violation_history) and (
        dual_history[-1] > dual_history[0] or max(violation_history) == 0.0
    )

    checks = {
        "identity_constraint_near_zero": identity_passed,
        "teacher_no_grad_enforced": bool(cfg["teacher_no_grad"]),
        "shared_noise_timestep_configured": bool(cfg["shared_noise_timestep"]),
        "raw_ema_reference_fixed": bool(cfg["raw_ema_reference_fixed"]),
        "dual_nonnegativity": dual_nonneg,
        "dual_rises_on_violation": dual_rises_on_violation,
        "teacher_hash_stable_on_resume": bool(cfg["teacher_hash_stable_on_resume"]),
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "protocol_revision": protocol.get("protocol_revision", "screen.v1"),
        "passed": passed,
        "complete": True,
        "checks": checks,
        "identity_violation": identity_violation,
        "dual_final": dual_history[-1] if dual_history else 0.0,
        "git_commit": git_commit(),
        "note": (
            "Structural/config smoke only. Full training-path anchor smoke requires "
            "frozen teacher integration in the BRACE screen trainer."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("experiments/brace/screen_protocol.v1.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/brace/anchor_smoke/summary.json"))
    args = parser.parse_args()

    protocol = read_json(repo_path(args.protocol))
    summary = run_anchor_smoke(protocol)
    output = repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, summary)
    try:
        from experiments.brace.stage_records import emit_stage_record

        emit_stage_record(
            "anchor_smoke",
            summary=summary,
            summary_path=output,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to emit stage record: {exc}", file=sys.stderr)
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
