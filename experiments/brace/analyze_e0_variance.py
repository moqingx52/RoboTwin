#!/usr/bin/env python3
"""E0 variance gate analysis: estimate Var_a[Q(s,a)] and the BRACE-RW bound.

Input: checks.jsonl produced by collect_e0_variance.py (one row per rollout,
fields: task, env_seed, snapshot_id, point_type, action_index, success).

Per branch point s (A actions x R continuations, one-way random effects):
  p_hat_a           = x_a / R                 (per-action success rate)
  V_hat(s)          = mean_a p_hat_a          (state value estimate)
  S2_between        = sum_a (p_hat_a - V_hat)^2 / (A - 1)
  S2_within         = mean_a x_a (R - x_a) / (R (R - 1))
  sigma2_Q_hat(s)   = S2_between - S2_within / R   (unbiased for Var_a[Q])
  Delta_hat_c(s)    = c * max(sigma2_Q_hat, 0) / (1 + c * V_hat)

Confirmatory: Tarone (1979) Z test for binomial overdispersion across the
pooled per-action counts (H0: Q(s,a) constant in a for each s => no
extra-binomial variation). One-sided; large Z => real action-level variance.

Gate (pre-registered in the protocol's e0_variance_gate.gate block):
  PASS if pooled mean Delta_hat_{c=c_gate} >= min_mean_delta
       and Tarone one-sided p <= tarone_alpha.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.replay_audit import (
    file_sha256,
    git_commit,
    read_json,
    repo_path,
    write_json_atomic,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def analyze_point(action_successes: dict[int, list[bool]], c_values: list[float]) -> dict[str, Any]:
    counts = {a: (sum(outcomes), len(outcomes)) for a, outcomes in sorted(action_successes.items())}
    replicates = {n for _, n in counts.values()}
    a_count = len(counts)
    r = min(replicates)
    p_hat = [x / n for x, n in counts.values()]
    v_hat = sum(p_hat) / a_count
    if a_count > 1:
        s2_between = sum((p - v_hat) ** 2 for p in p_hat) / (a_count - 1)
    else:
        s2_between = 0.0
    if r > 1:
        s2_within = sum(x * (n - x) / (n * (n - 1)) for x, n in counts.values()) / a_count
        sigma2_q = s2_between - s2_within / r
    else:
        s2_within = float("nan")
        sigma2_q = float("nan")
    sigma2_q_clipped = max(sigma2_q, 0.0) if not math.isnan(sigma2_q) else float("nan")
    deltas = {
        f"delta_c{c:g}": (c * sigma2_q_clipped / (1.0 + c * v_hat)) if not math.isnan(sigma2_q) else None
        for c in c_values
    }
    return {
        "actions": a_count,
        "replicates": r,
        "balanced": len(replicates) == 1,
        "per_action_successes": [x for x, _ in counts.values()],
        "v_hat": v_hat,
        "s2_between": s2_between,
        "s2_within": None if math.isnan(s2_within) else s2_within,
        "sigma2_q_hat": None if math.isnan(sigma2_q) else sigma2_q,
        "sigma2_q_hat_clipped": None if math.isnan(sigma2_q) else sigma2_q_clipped,
        **deltas,
    }


def tarone_z(groups: list[list[tuple[int, int]]]) -> dict[str, Any]:
    """Tarone (1979) one-sided Z test for extra-binomial variation.

    Each group is one branch point: a list of (x_a, n_a) per-action counts.
    H0 within a group: all actions share that group's success probability.
    The statistic pools per-group chi-square-like terms; groups with
    p_hat in {0, 1} carry no information and are skipped.
    """
    s_stat = 0.0
    n_total = 0.0
    var_term = 0.0
    informative_groups = 0
    for group in groups:
        x_sum = sum(x for x, _ in group)
        n_sum = sum(n for _, n in group)
        if n_sum == 0:
            continue
        p = x_sum / n_sum
        if p <= 0.0 or p >= 1.0:
            continue
        informative_groups += 1
        s_stat += sum((x - n * p) ** 2 for x, n in group) / (p * (1.0 - p))
        n_total += n_sum
        var_term += sum(n * (n - 1) for _, n in group)
    if var_term <= 0:
        return {"z": None, "p_one_sided": None, "informative_points": informative_groups}
    z = (s_stat - n_total) / math.sqrt(2.0 * var_term)
    return {
        "z": z,
        "p_one_sided": normal_sf(z),
        "informative_points": informative_groups,
    }


def bootstrap_ci(values: list[float], *, iterations: int, seed: int, alpha: float = 0.05) -> dict[str, float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(alpha / 2 * iterations)]
    hi = means[min(int((1 - alpha / 2) * iterations), iterations - 1)]
    return {"lower": lo, "upper": hi, "alpha": alpha, "iterations": iterations}


def summarize(points: list[dict[str, Any]], c_values: list[float], *, bootstrap_seed: int) -> dict[str, Any]:
    def mean(key: str) -> float | None:
        vals = [p[key] for p in points if p.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    out: dict[str, Any] = {
        "points": len(points),
        "mean_v_hat": mean("v_hat"),
        "mean_sigma2_q_hat": mean("sigma2_q_hat"),
        "mean_sigma2_q_hat_clipped": mean("sigma2_q_hat_clipped"),
    }
    for c in c_values:
        key = f"delta_c{c:g}"
        out[f"mean_{key}"] = mean(key)
        vals = [p[key] for p in points if p.get(key) is not None]
        out[f"mean_{key}_ci95"] = bootstrap_ci(vals, iterations=10000, seed=bootstrap_seed)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="collect_e0_variance.py output dir")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="default: <input-dir>/analysis.json")
    args = parser.parse_args()

    input_dir = repo_path(args.input_dir)
    protocol_path = repo_path(args.protocol)
    protocol = read_json(protocol_path)
    e0 = protocol["e0_variance_gate"]
    c_values = [float(c) for c in e0.get("c_values", [1.0, 3.0])]
    gate_cfg = e0.get("gate", {})
    c_gate = float(gate_cfg.get("c", 3.0))
    min_mean_delta = float(gate_cfg.get("min_mean_delta", 0.02))
    tarone_alpha = float(gate_cfg.get("tarone_alpha", 0.05))
    bootstrap_seed = int(e0.get("bootstrap_seed", 20260814))

    rows = read_jsonl(input_dir / "checks.jsonl")
    if not rows:
        print("No rows in checks.jsonl", file=sys.stderr)
        return 1

    grouped: dict[tuple[str, int, int], dict[int, list[bool]]] = defaultdict(lambda: defaultdict(list))
    meta: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["task"]), int(row["env_seed"]), int(row["snapshot_id"]))
        grouped[key][int(row["action_index"])].append(bool(row["success"]))
        meta[key] = {"point_type": row.get("point_type"), "branch_chunk_index": row.get("branch_chunk_index")}

    point_results = []
    for key in sorted(grouped):
        task, env_seed, snapshot_id = key
        result = analyze_point(grouped[key], c_values)
        point_results.append(
            {"task": task, "env_seed": env_seed, "snapshot_id": snapshot_id, **meta[key], **result}
        )

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in point_results:
        by_type[str(point.get("point_type"))].append(point)
        by_task[point["task"]].append(point)

    tarone_groups = [
        [(x, p["replicates"]) for x in p["per_action_successes"]]
        for p in point_results
        if p["balanced"]
    ]
    tarone = tarone_z(tarone_groups)

    pooled = summarize(point_results, c_values, bootstrap_seed=bootstrap_seed)
    gate_key = f"mean_delta_c{c_gate:g}"
    gate_delta = pooled.get(gate_key)
    tarone_pass = tarone["p_one_sided"] is not None and tarone["p_one_sided"] <= tarone_alpha
    delta_pass = gate_delta is not None and gate_delta >= min_mean_delta
    gate_passed = bool(tarone_pass and delta_pass)

    analysis = {
        "schema_version": 1,
        "stage": "e0_variance_gate_analysis",
        "protocol_revision": protocol.get("protocol_revision"),
        "protocol_path": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "git_commit": git_commit(),
        "inputs": {
            "checks": str(input_dir / "checks.jsonl"),
            "rollouts": len(rows),
            "points": len(point_results),
        },
        "estimator": "sigma2_Q = S2_between - S2_within / R (one-way random effects, unbiased)",
        "pooled": pooled,
        "by_point_type": {name: summarize(pts, c_values, bootstrap_seed=bootstrap_seed) for name, pts in sorted(by_type.items())},
        "by_task": {name: summarize(pts, c_values, bootstrap_seed=bootstrap_seed) for name, pts in sorted(by_task.items())},
        "tarone_overdispersion": {**tarone, "alpha": tarone_alpha, "passed": tarone_pass},
        "gate": {
            "c": c_gate,
            "min_mean_delta": min_mean_delta,
            "observed_mean_delta": gate_delta,
            "delta_criterion_passed": delta_pass,
            "tarone_criterion_passed": tarone_pass,
            "e0_gate_passed": gate_passed,
        },
        "points": point_results,
    }

    output_path = repo_path(args.output) if args.output else input_dir / "analysis.json"
    write_json_atomic(output_path, analysis)

    print(f"E0 analysis: {len(point_results)} points, {len(rows)} rollouts")
    print(f"  pooled V_hat            = {pooled['mean_v_hat']:.4f}")
    sigma = pooled["mean_sigma2_q_hat"]
    print(f"  pooled sigma2_Q_hat     = {sigma:.5f}" if sigma is not None else "  pooled sigma2_Q_hat     = n/a")
    for c in c_values:
        val = pooled.get(f"mean_delta_c{c:g}")
        ci = pooled.get(f"mean_delta_c{c:g}_ci95")
        ci_str = f" (95% CI [{ci['lower']:.4f}, {ci['upper']:.4f}])" if ci else ""
        print(f"  mean Delta_hat (c={c:g})  = {val:.4f}{ci_str}" if val is not None else f"  mean Delta_hat (c={c:g})  = n/a")
    if tarone["z"] is not None:
        print(f"  Tarone Z = {tarone['z']:.3f}, one-sided p = {tarone['p_one_sided']:.2e}")
    print(f"  E0 GATE: {'PASS' if gate_passed else 'FAIL'} "
          f"(Delta_c{c_gate:g} >= {min_mean_delta}: {delta_pass}; Tarone p <= {tarone_alpha}: {tarone_pass})")
    print(f"  analysis -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
