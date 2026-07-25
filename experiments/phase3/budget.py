#!/usr/bin/env python3
"""Matched-budget accounting for Phase 3 CPST experiments.

Three axes (all runs should record):
  - interaction: rollout attempts / env transitions
  - accepted_data: successful trajectories and non-expert action chunks by source
  - training: optimizer steps, batches, loss mass by source
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from common import atomic_write_json, now, read_json


@dataclass
class InteractionBudget:
    rollout_attempts: int = 0
    env_transitions: int = 0
    guided_rollout_attempts: int = 0


@dataclass
class AcceptedDataBudget:
    expert_chunks: int = 0
    rollout_success_chunks: int = 0
    prefix_chunks: int = 0
    success_trajectories: int = 0
    prefix_trajectories: int = 0

    @property
    def non_expert_chunks(self) -> int:
        return self.rollout_success_chunks + self.prefix_chunks


@dataclass
class TrainingBudget:
    optimizer_steps: int = 0
    batches: int = 0
    loss_mass_expert: float = 0.0
    loss_mass_rollout: float = 0.0
    loss_mass_prefix: float = 0.0
    seen_chunks_expert: int = 0
    seen_chunks_rollout: int = 0
    seen_chunks_prefix: int = 0

    @property
    def loss_mass_non_expert(self) -> float:
        return self.loss_mass_rollout + self.loss_mass_prefix


@dataclass
class BudgetManifest:
    task: str
    main_id: str
    absorption_id: str
    train_seed: int
    iteration: int = 0
    interaction: InteractionBudget = field(default_factory=InteractionBudget)
    accepted_data: AcceptedDataBudget = field(default_factory=AcceptedDataBudget)
    training: TrainingBudget = field(default_factory=TrainingBudget)
    created_at: str = field(default_factory=now)
    updated_at: str = field(default_factory=now)
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "BudgetManifest":
        return cls(
            task=payload["task"],
            main_id=payload["main_id"],
            absorption_id=payload["absorption_id"],
            train_seed=int(payload["train_seed"]),
            iteration=int(payload.get("iteration", 0)),
            interaction=InteractionBudget(**payload.get("interaction", {})),
            accepted_data=AcceptedDataBudget(**payload.get("accepted_data", {})),
            training=TrainingBudget(**payload.get("training", {})),
            created_at=payload.get("created_at", now()),
            updated_at=payload.get("updated_at", now()),
            notes=payload.get("notes", {}),
        )


def load_manifest(path: Path) -> BudgetManifest:
    return BudgetManifest.from_dict(read_json(path))


def save_manifest(path: Path, manifest: BudgetManifest) -> None:
    manifest.updated_at = now()
    atomic_write_json(path, manifest.to_dict())


def record_training_step(
    manifest: BudgetManifest,
    *,
    expert_count: int,
    rollout_count: int,
    prefix_count: int,
    weighted_expert_contrib: float,
    weighted_rollout_contrib: float,
    weighted_prefix_contrib: float = 0.0,
) -> None:
    manifest.training.optimizer_steps += 1
    manifest.training.batches += 1
    manifest.training.seen_chunks_expert += expert_count
    manifest.training.seen_chunks_rollout += rollout_count
    manifest.training.seen_chunks_prefix += prefix_count
    manifest.training.loss_mass_expert += weighted_expert_contrib
    manifest.training.loss_mass_rollout += weighted_rollout_contrib
    manifest.training.loss_mass_prefix += weighted_prefix_contrib


def summarize_manifests(manifests: list[BudgetManifest]) -> dict:
    if not manifests:
        return {}
    return {
        "count": len(manifests),
        "interaction_rollout_attempts": [m.interaction.rollout_attempts for m in manifests],
        "accepted_non_expert_chunks": [m.accepted_data.non_expert_chunks for m in manifests],
        "training_steps": [m.training.optimizer_steps for m in manifests],
        "loss_mass_non_expert": [m.training.loss_mass_non_expert for m in manifests],
    }
