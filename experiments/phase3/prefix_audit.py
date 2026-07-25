#!/usr/bin/env python3
"""Precision audit for verified failure-prefix extraction.

Tests the asymmetric hypothesis that failed trajectories contain reusable
successful subsegments. Does NOT claim to fix bad-in-success credit leakage.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from common import PHASE3, atomic_write_json, now, read_json


@dataclass
class PrefixCandidate:
    task: str
    env_seed: int
    rollout_id: int
    hdf5_path: str
    total_steps: int
    prefix_steps: int
    prefix_fraction: float
    extraction_mode: str
    j_hat: float = 0.0

    @property
    def prefix_ratio(self) -> float:
        if self.total_steps <= 0:
            return 0.0
        return self.prefix_steps / self.total_steps


@dataclass
class PrefixAuditReport:
    task: str
    extraction_mode: str
    num_candidates: int
    prefix_steps: dict
    prefix_fractions: dict
    audit_passed: bool
    checks: dict = field(default_factory=dict)
    created_at: str = field(default_factory=now)
    samples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p50": float(np.percentile(arr, 50)),
    }


def load_prefix_candidates(task: str, candidates_path: Path) -> list[PrefixCandidate]:
    payload = read_json(candidates_path)
    return [PrefixCandidate(**row) for row in payload["candidates"]]


def run_audit(
    task: str,
    candidates: list[PrefixCandidate],
    *,
    min_prefix_steps: int = 5,
    min_prefix_fraction: float = 0.1,
    max_prefix_fraction: float = 0.95,
    min_candidates: int = 10,
) -> PrefixAuditReport:
    steps = [c.prefix_steps for c in candidates]
    fracs = [c.prefix_fraction for c in candidates]

    checks = {
        "min_candidates": len(candidates) >= min_candidates,
        "prefix_steps_positive": all(s >= min_prefix_steps for s in steps) if steps else False,
        "prefix_fraction_in_range": all(
            min_prefix_fraction <= f <= max_prefix_fraction for f in fracs
        )
        if fracs
        else False,
        # Placeholders for follow-up audits (require sim continue-rollout).
        "continue_rollout_evaluated": False,
        "subgoal_verified": False,
        "negative_controls_run": False,
    }

    mode = candidates[0].extraction_mode if candidates else "unknown"
    samples = [asdict(c) for c in random.sample(candidates, min(20, len(candidates)))]

    report = PrefixAuditReport(
        task=task,
        extraction_mode=mode,
        num_candidates=len(candidates),
        prefix_steps=_stats([float(s) for s in steps]),
        prefix_fractions=_stats([float(f) for f in fracs]),
        audit_passed=all(checks[k] for k in ("min_candidates", "prefix_steps_positive", "prefix_fraction_in_range")),
        checks=checks,
        samples=samples,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="Audit verified failure-prefix candidates.")
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="JSON from build_prefix_dataset.py --write-candidates",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-candidates", type=int, default=10)
    args = parser.parse_args()

    candidates_path = args.candidates or (PHASE3 / "prefix_candidates" / f"{args.task}.json")
    if not candidates_path.is_file():
        raise SystemExit(f"Missing candidates file: {candidates_path}. Run build_prefix_dataset.py first.")

    candidates = load_prefix_candidates(args.task, candidates_path)
    report = run_audit(args.task, candidates, min_candidates=args.min_candidates)

    output = args.output or (PHASE3 / "prefix_audit" / f"{args.task}.json")
    atomic_write_json(output, report.to_dict())

    status = "PASSED" if report.audit_passed else "FAILED"
    print(f"[{args.task}] prefix audit {status}: {output}")
    if not report.audit_passed:
        print("Failed checks:", {k: v for k, v in report.checks.items() if not v})
        raise SystemExit(1)


if __name__ == "__main__":
    main()
