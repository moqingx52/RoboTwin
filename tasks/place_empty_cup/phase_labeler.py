from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


PHASE_NAMES = (
    "approach cup",
    "grasp",
    "lift",
    "move",
    "place",
    "release",
)


@dataclass
class PhaseLabel:
    phase_id: int
    phase_name: str


def _arm_delta(action: np.ndarray) -> float:
    left = np.linalg.norm(action[:6])
    right = np.linalg.norm(action[7:13])
    return float(max(left, right))


def _gripper_closed(action: np.ndarray) -> bool:
    return bool(action[6] < 0.35 or action[13] < 0.35)


def _gripper_open(action: np.ndarray) -> bool:
    return bool(action[6] > 0.75 and action[13] > 0.75)


def infer_place_empty_cup_phase(
    action_t: np.ndarray,
    step_idx: int,
    total_steps: int,
) -> PhaseLabel:
    progress = 0.0 if total_steps <= 1 else float(step_idx) / float(total_steps - 1)
    move_mag = _arm_delta(action_t)
    closed = _gripper_closed(action_t)
    opened = _gripper_open(action_t)

    if progress < 0.18 and not closed:
        return PhaseLabel(0, PHASE_NAMES[0])
    if closed and progress < 0.38:
        return PhaseLabel(1, PHASE_NAMES[1])
    if closed and move_mag > 0.05 and progress < 0.55:
        return PhaseLabel(2, PHASE_NAMES[2])
    if closed and progress < 0.78:
        return PhaseLabel(3, PHASE_NAMES[3])
    if not opened and progress < 0.92:
        return PhaseLabel(4, PHASE_NAMES[4])
    return PhaseLabel(5, PHASE_NAMES[5])


def label_sequence(actions: Iterable[np.ndarray]) -> list[PhaseLabel]:
    action_list = [np.asarray(a, dtype=np.float32) for a in actions]
    total = len(action_list)
    return [infer_place_empty_cup_phase(a, i, total) for i, a in enumerate(action_list)]
