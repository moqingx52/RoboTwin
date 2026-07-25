#!/usr/bin/env python3
"""Coverage-aware model selection for Phase 3 CPST."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase2b"))

from promotion import (  # noqa: E402
    HARD_GAIN_THRESHOLDS,
    evaluate_full,
    evaluate_screen,
    load_base_metrics,
)

__all__ = [
    "HARD_GAIN_THRESHOLDS",
    "evaluate_full",
    "evaluate_screen",
    "load_base_metrics",
    "evaluate_coverage_aware_selection",
]


def evaluate_coverage_aware_selection(task, eval_payload, base_metrics, previous_payload=None):
    """Screen gate + explicit naming for the evaluation protocol (not a training algorithm)."""
    result = evaluate_screen(task, eval_payload, base_metrics, previous_payload)
    result["gate"] = "coverage_aware_model_selection"
    result["protocol"] = (
        "Select checkpoints with id_mean_sr >= 80% base, id_coverage >= 80% base, "
        "train_mean_sr >= 75% base, then maximize hard_20 under those constraints."
    )
    return result
