#!/usr/bin/env python3
"""Promotion gates for Phase 2b screen and full evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import read_json

HARD_GAIN_THRESHOLDS = {
    "place_container_plate": {"mean_sr": 0.10, "coverage": 0.25},
    "dump_bin_bigbin": {"mean_sr": 0.05, "coverage": 0.15},
}


def _metric(payload, split, metric):
    return float(payload["splits"][split][metric])


def load_base_metrics(task, base_path):
    payload = read_json(base_path)
    return {
        "id_mean_sr": _metric(payload, "id_heldout", "mean_sr"),
        "id_coverage": _metric(payload, "id_heldout", "solved_coverage"),
        "train_mean_sr": _metric(payload, "train_seen", "mean_sr"),
        "train_coverage": _metric(payload, "train_seen", "solved_coverage"),
        "hard_mean_sr": _metric(payload, "hard_20", "mean_sr"),
        "hard_coverage": _metric(payload, "hard_20", "solved_coverage"),
    }


def evaluate_screen(task, eval_payload, base_metrics, previous_payload=None):
    metrics = {
        "id_mean_sr": _metric(eval_payload, "id_heldout", "mean_sr"),
        "id_coverage": _metric(eval_payload, "id_heldout", "solved_coverage"),
        "train_mean_sr": _metric(eval_payload, "train_seen", "mean_sr"),
        "train_coverage": _metric(eval_payload, "train_seen", "solved_coverage"),
    }
    reasons = []
    passed = True

    if metrics["id_mean_sr"] < base_metrics["id_mean_sr"] * 0.80:
        passed = False
        reasons.append("id_mean_sr below 80% of base")
    if metrics["id_coverage"] < base_metrics["id_coverage"] * 0.80:
        passed = False
        reasons.append("id_coverage below 80% of base")
    if metrics["train_mean_sr"] < base_metrics["train_mean_sr"] * 0.75:
        passed = False
        reasons.append("train_mean_sr below 75% of base")

    if previous_payload is not None:
        prev_id = _metric(previous_payload, "id_heldout", "mean_sr")
        if metrics["id_mean_sr"] < prev_id - 0.10:
            passed = False
            reasons.append("id_mean_sr dropped more than 10pp vs previous checkpoint")

    return {
        "passed": passed,
        "metrics": metrics,
        "base_metrics": base_metrics,
        "reasons": reasons,
        "gate": "screen",
        "task": task,
    }


def evaluate_full(task, eval_payload, base_metrics):
    metrics = {
        "id_mean_sr": _metric(eval_payload, "id_heldout", "mean_sr"),
        "id_coverage": _metric(eval_payload, "id_heldout", "solved_coverage"),
        "train_mean_sr": _metric(eval_payload, "train_seen", "mean_sr"),
        "train_coverage": _metric(eval_payload, "train_seen", "solved_coverage"),
        "hard_mean_sr": _metric(eval_payload, "hard_20", "mean_sr"),
        "hard_coverage": _metric(eval_payload, "hard_20", "solved_coverage"),
    }
    reasons = []
    passed = True

    if metrics["id_mean_sr"] < base_metrics["id_mean_sr"] * 0.90:
        passed = False
        reasons.append("id_mean_sr below 90% of base")
    if metrics["train_mean_sr"] < base_metrics["train_mean_sr"] * 0.85:
        passed = False
        reasons.append("train_mean_sr below 85% of base")
    if metrics["id_coverage"] < base_metrics["id_coverage"] * 0.90:
        passed = False
        reasons.append("id_coverage below 90% of base")

    hard_threshold = HARD_GAIN_THRESHOLDS[task]
    hard_ok = (
        metrics["hard_mean_sr"] >= hard_threshold["mean_sr"]
        or metrics["hard_coverage"] >= hard_threshold["coverage"]
    )
    if not hard_ok:
        passed = False
        reasons.append("hard_20 gain below task threshold")

    return {
        "passed": passed,
        "metrics": metrics,
        "base_metrics": base_metrics,
        "reasons": reasons,
        "gate": "full",
        "task": task,
    }


def evaluate_diagnose(task, a0_payload, a2_payload):
    a0_id = _metric(a0_payload, "id_heldout", "mean_sr")
    a2_id = _metric(a2_payload, "id_heldout", "mean_sr")
    ratio = a2_id / a0_id if a0_id > 0 else 0.0
    normalizer_is_primary = ratio < 0.80
    return {
        "task": task,
        "a0_id_mean_sr": a0_id,
        "a2_id_mean_sr": a2_id,
        "a2_over_a0_ratio": ratio,
        "normalizer_is_primary_cause": normalizer_is_primary,
        "conclusion": (
            "normalizer replacement is likely the primary cause"
            if normalizer_is_primary
            else "training-time rollout gradients are more likely than epoch-0 normalizer swap"
        ),
    }


def rank_screen_winners(screen_results):
    """Return sorted list of (task, candidate, epoch, score) for full eval."""
    ranked = []
    for key, result in screen_results.items():
        if not result.get("passed"):
            continue
        metrics = result["metrics"]
        score = metrics["id_mean_sr"] + 0.5 * metrics["train_mean_sr"]
        task, candidate, epoch = key.split(":")
        ranked.append((score, task, candidate, int(epoch), result))
    ranked.sort(reverse=True)
    return ranked
