#!/usr/bin/env python3
"""Manifest-level chunk dataset loader for BRACE Screen 1 SFT."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from experiments.brace.control_trace import load_brace_trace, policy_chunk_actions


@dataclass(frozen=True)
class ChunkExample:
    dataset: str
    task: str
    env_seed: int
    branch_chunk_index: int
    hdf5_path: Path
    actions: np.ndarray
    metadata: dict[str, Any]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_chunk_actions(record: dict[str, Any]) -> np.ndarray:
    trace = load_brace_trace(Path(record["hdf5_path"]))
    chunk_index = int(record["branch_chunk_index"])
    chunk = next(item for item in trace["policy_chunks"] if int(item["chunk_index"]) == chunk_index)
    return policy_chunk_actions(chunk["action"])


def iter_chunk_examples(manifest_path: Path) -> Iterator[ChunkExample]:
    for record in read_jsonl(manifest_path):
        actions = load_chunk_actions(record)
        yield ChunkExample(
            dataset=str(record["dataset"]),
            task=str(record["task"]),
            env_seed=int(record["env_seed"]),
            branch_chunk_index=int(record["branch_chunk_index"]),
            hdf5_path=Path(record["hdf5_path"]),
            actions=actions,
            metadata=record,
        )


def load_manifest_dataset(manifest_path: Path) -> list[ChunkExample]:
    return list(iter_chunk_examples(manifest_path))


def paired_b1_n1_paths(dataset_dir: Path, run_label: str) -> tuple[Path, Path]:
    return (
        dataset_dir / f"{run_label}_B1.jsonl",
        dataset_dir / f"{run_label}_N1.jsonl",
    )
