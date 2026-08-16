"""Preservation-group stratified batch sampling for BRACE anchor replay."""

from __future__ import annotations

import numpy as np


class PreservationGroupBatchSampler:
    def __init__(
        self,
        preservation_groups: np.ndarray,
        batch_size: int,
        preservation_group_ids: dict[str, int],
        samples_per_group: int,
        *,
        seed: int = 0,
        num_batches: int | None = None,
        allowed_indices: np.ndarray | None = None,
    ):
        if preservation_groups is None or len(preservation_groups) == 0:
            raise ValueError("preservation_stratified sampler requires preservation group labels")
        self.batch_size = int(batch_size)
        self.samples_per_group = int(samples_per_group)
        required = len(preservation_group_ids) * self.samples_per_group
        if required != self.batch_size:
            raise ValueError(
                f"anchor batch_size={self.batch_size} must equal "
                f"len(groups)*samples_per_group={required}"
            )
        self.group_ids = preservation_group_ids
        self.indices_by_group: dict[str, np.ndarray] = {}
        for name, group_id in preservation_group_ids.items():
            indices = np.flatnonzero(np.asarray(preservation_groups) == int(group_id))
            if indices.size == 0:
                raise ValueError(f"anchor replay set missing preservation group {name} (id={group_id})")
            self.indices_by_group[name] = indices.astype(np.int64, copy=False)
        if allowed_indices is not None:
            allowed = np.asarray(allowed_indices, dtype=np.int64)
            for name in self.group_ids:
                pool = self.indices_by_group[name]
                filtered = np.intersect1d(pool, allowed, assume_unique=False)
                if filtered.size == 0:
                    raise ValueError(f"anchor split left no samples for preservation group {name}")
                self.indices_by_group[name] = filtered.astype(np.int64, copy=False)
        self.rng = np.random.default_rng(seed)
        self.num_batch = (
            int(num_batches)
            if num_batches is not None
            else max(1, min(len(v) for v in self.indices_by_group.values()))
        )

    def __iter__(self):
        for _ in range(self.num_batch):
            picks = []
            for name in self.group_ids:
                pool = self.indices_by_group[name]
                picks.append(
                    self.rng.choice(
                        pool,
                        size=self.samples_per_group,
                        replace=pool.size < self.samples_per_group,
                    )
                )
            batch = np.concatenate(picks).astype(np.int64, copy=False)
            self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batch
