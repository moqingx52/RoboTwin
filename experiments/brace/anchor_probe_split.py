"""Held-out anchor probe split by env_seed for diagnostic evaluation."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AnchorProbeSplit:
    train_env_seeds: frozenset[int]
    probe_env_seeds: frozenset[int]
    train_episode_indices: tuple[int, ...]
    probe_episode_indices: tuple[int, ...]
    manifest_sha256: str
    split_seed: int
    holdout_fraction: float

    @property
    def split_sha256(self) -> str:
        payload = {
            "train_env_seeds": sorted(self.train_env_seeds),
            "probe_env_seeds": sorted(self.probe_env_seeds),
            "manifest_sha256": self.manifest_sha256,
            "split_seed": self.split_seed,
            "holdout_fraction": self.holdout_fraction,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return digest

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_env_seeds": sorted(self.train_env_seeds),
            "probe_env_seeds": sorted(self.probe_env_seeds),
            "train_episode_indices": list(self.train_episode_indices),
            "probe_episode_indices": list(self.probe_episode_indices),
            "manifest_sha256": self.manifest_sha256,
            "split_seed": self.split_seed,
            "holdout_fraction": self.holdout_fraction,
            "split_sha256": self.split_sha256,
        }


def build_anchor_probe_split(
    manifest: dict[str, Any],
    *,
    split_seed: int = 42,
    holdout_fraction: float = 0.5,
    min_probe_seeds_per_group: int = 1,
) -> AnchorProbeSplit:
    episodes = manifest.get("episodes", [])
    if not episodes:
        raise ValueError("anchor manifest has no episodes")
    manifest_sha256 = str(manifest.get("manifest_sha256", ""))
    by_group: dict[str, list[tuple[int, int]]] = {}
    for episode_index, row in enumerate(episodes):
        group = str(row["preservation_group"])
        by_group.setdefault(group, []).append((episode_index, int(row["env_seed"])))

    rng = random.Random(split_seed)
    train_episode_indices: list[int] = []
    probe_episode_indices: list[int] = []
    train_env_seeds: set[int] = set()
    probe_env_seeds: set[int] = set()
    for group, rows in sorted(by_group.items()):
        unique_seeds = sorted({env_seed for _, env_seed in rows})
        shuffled = list(unique_seeds)
        rng.shuffle(shuffled)
        n_probe = max(min_probe_seeds_per_group, int(round(len(shuffled) * holdout_fraction)))
        n_probe = min(n_probe, len(shuffled) - min_probe_seeds_per_group) if len(shuffled) > min_probe_seeds_per_group else min(n_probe, len(shuffled))
        probe_seed_set = set(shuffled[:n_probe])
        for episode_index, env_seed in rows:
            if env_seed in probe_seed_set:
                probe_episode_indices.append(episode_index)
                probe_env_seeds.add(env_seed)
            else:
                train_episode_indices.append(episode_index)
                train_env_seeds.add(env_seed)

    if not probe_env_seeds or not train_env_seeds:
        raise ValueError(
            f"degenerate anchor probe split: train_seeds={len(train_env_seeds)} probe_seeds={len(probe_env_seeds)}"
        )
    if train_env_seeds & probe_env_seeds:
        raise ValueError("anchor probe split must be disjoint by env_seed")

    return AnchorProbeSplit(
        train_env_seeds=frozenset(train_env_seeds),
        probe_env_seeds=frozenset(probe_env_seeds),
        train_episode_indices=tuple(sorted(train_episode_indices)),
        probe_episode_indices=tuple(sorted(probe_episode_indices)),
        manifest_sha256=manifest_sha256,
        split_seed=split_seed,
        holdout_fraction=holdout_fraction,
    )


def sequence_indices_for_env_seeds(dataset, env_seeds: set[int] | frozenset[int]) -> np.ndarray:
    if getattr(dataset, "sample_env_seeds", None) is None:
        raise ValueError("anchor dataset missing sample_env_seeds metadata")
    allowed = np.asarray(sorted(env_seeds), dtype=np.int64)
    return np.flatnonzero(np.isin(dataset.sample_env_seeds, allowed)).astype(np.int64, copy=False)


def build_fixed_stratified_batch(
    dataset,
    allowed_indices: np.ndarray,
    preservation_group_ids: dict[str, int],
    samples_per_group: int,
    *,
    seed: int,
) -> np.ndarray:
    if allowed_indices.size == 0:
        raise ValueError("cannot build stratified batch from empty allowed_indices")
    groups = getattr(dataset, "sample_preservation_groups", None)
    if groups is None:
        raise ValueError("dataset missing sample_preservation_groups")
    rng = np.random.default_rng(seed)
    picks: list[np.ndarray] = []
    for group_name, group_id in preservation_group_ids.items():
        pool = allowed_indices[groups[allowed_indices] == int(group_id)]
        if pool.size == 0:
            raise ValueError(f"probe split missing preservation group {group_name}")
        picks.append(
            rng.choice(pool, size=samples_per_group, replace=pool.size < samples_per_group)
        )
    batch = np.concatenate(picks).astype(np.int64, copy=False)
    rng.shuffle(batch)
    return batch
