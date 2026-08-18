#!/usr/bin/env python3
"""Shared T2 training helpers: frozen launch-config I/O, mixture paths, rho math."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from common import read_json, repo_path

CT_DIR = Path(__file__).resolve().parent
LAUNCH_CONFIG_FILE = CT_DIR / "t2_launch_config.json"
PROTOCOL_FILE = CT_DIR / "protocol.t2_training.v1.json"
DEFAULT_MIXTURE_DIR = CT_DIR / "mixtures"

# Frozen at protocol freeze; verified against the companion file at load time.
EXPECTED_LAUNCH_CONFIG_SHA256 = "cc89f91e35c066fe6ca7768ba5487505281b6fe20c651dc03178c3e2eb19fede"
EXPECTED_PROTOCOL_SHA256 = "5c6c774b2c26d272146d8d724bf19366de2641866929dbdad42266d13dd5b34e"

SELF_DIVERSE_NAME_RE = re.compile(r"episode_(\d+)_a(\d+)\.hdf5$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_companion_sha256(path: Path) -> str:
    companion = path.with_name(path.name + ".sha256")
    if not companion.exists():
        raise SystemExit(f"missing sha256 companion for frozen input: {companion}")
    recorded = companion.read_text().split()[0]
    actual = sha256_file(path)
    if recorded != actual:
        raise SystemExit(f"sha256 mismatch for {path}: recorded {recorded}, actual {actual}")
    return actual


def load_launch_config() -> Tuple[Dict, str]:
    sha = verify_companion_sha256(LAUNCH_CONFIG_FILE)
    if sha != EXPECTED_LAUNCH_CONFIG_SHA256:
        raise SystemExit(
            f"t2_launch_config.json sha256 {sha} does not match the freeze-time "
            f"value {EXPECTED_LAUNCH_CONFIG_SHA256}"
        )
    protocol_sha = verify_companion_sha256(PROTOCOL_FILE)
    if protocol_sha != EXPECTED_PROTOCOL_SHA256:
        raise SystemExit(
            f"protocol.t2_training.v1.json sha256 {protocol_sha} does not match "
            f"the freeze-time value {EXPECTED_PROTOCOL_SHA256}"
        )
    cfg = read_json(LAUNCH_CONFIG_FILE)
    recorded = cfg["depends_on"]["protocol.t2_training.v1.json"]
    if recorded != protocol_sha:
        raise SystemExit(
            f"launch_config depends_on protocol sha {recorded} != file {protocol_sha}"
        )
    return cfg, sha


def job_id(point: str, seed: int) -> str:
    return f"{point}_seed{int(seed):02d}"


def mixture_zarr_path(point: str, mixture_dir: Path = DEFAULT_MIXTURE_DIR) -> Path:
    if point == "Zero":
        return repo_path("policy", "DP", "data", "place_container_plate-demo_clean-200.zarr")
    safe = point.replace(" ", "_")
    return Path(mixture_dir) / f"place_container_plate.{safe}.zarr"


def parse_self_diverse_episode(rel_path: str) -> Tuple[int, int]:
    name = Path(rel_path).name
    match = SELF_DIVERSE_NAME_RE.search(name)
    if not match:
        raise ValueError(f"cannot parse Self-Diverse hdf5 name: {rel_path}")
    return int(match.group(1)), int(match.group(2))


def resolve_new_source_episodes(point: str, mixture_cfg: Dict) -> List[Dict]:
    """Return ordered new-source episode records for one training point."""
    if point == "Zero" or not mixture_cfg.get("Q"):
        return []
    source_run = repo_path(mixture_cfg["source_run_dir"])
    records: List[Dict] = []
    if "new_source_hdf5" in mixture_cfg:
        for rel in mixture_cfg["new_source_hdf5"]:
            path = source_run / rel
            env_seed, attempt_index = parse_self_diverse_episode(rel)
            records.append(
                {
                    "path": path,
                    "relpath": rel,
                    "env_seed": env_seed,
                    "attempt_index": attempt_index,
                }
            )
        return records
    pattern = mixture_cfg["new_source_hdf5_pattern"]
    for env_seed in mixture_cfg["new_source_seeds"]:
        rel = pattern.format(seed=int(env_seed))
        records.append(
            {
                "path": source_run / rel,
                "relpath": rel,
                "env_seed": int(env_seed),
                "attempt_index": 0,
            }
        )
    return records


def missing_new_source_paths(point: str, mixture_cfg: Dict) -> List[str]:
    missing = []
    for rec in resolve_new_source_episodes(point, mixture_cfg):
        if not rec["path"].is_file():
            missing.append(str(rec["path"]))
    return missing


def expert_ratio_from_rho(rho: float) -> Optional[float]:
    """DP expert_ratio is the Base200 (source 0) fraction = 1 - rho."""
    if rho <= 0:
        return None
    return 1.0 - float(rho)


def realized_batch_mix(batch_size: int, rho: float) -> Dict:
    if rho <= 0:
        return {
            "rho_target": 0.0,
            "expert_ratio": None,
            "expert_per_batch": int(batch_size),
            "rollout_per_batch": 0,
            "rho_realized": 0.0,
        }
    expert_ratio = expert_ratio_from_rho(rho)
    expert_per_batch = int(round(batch_size * float(expert_ratio)))
    rollout_per_batch = int(batch_size) - expert_per_batch
    return {
        "rho_target": float(rho),
        "expert_ratio": float(expert_ratio),
        "expert_per_batch": expert_per_batch,
        "rollout_per_batch": rollout_per_batch,
        "rho_realized": rollout_per_batch / float(batch_size),
    }


def w_traj_from_rho(rho: float, q: int) -> Optional[float]:
    if q <= 0 or rho <= 0:
        return None
    return (200.0 * float(rho)) / (float(q) * (1.0 - float(rho)))


def iter_jobs(cfg: Dict) -> List[Tuple[str, int]]:
    points = list(cfg["training_grid"]["points"])
    seeds = [int(s) for s in cfg["training_grid"]["paired_training_seeds"]]
    return [(point, seed) for point in points for seed in seeds]


def checkpoint_path(job_dir: Path, seed: int) -> Path:
    return Path(job_dir) / "checkpoints" / f"t2-{int(seed)}" / "1.ckpt"


def python_bin() -> str:
    conda = Path("/root/miniconda/envs/RoboTwin/bin/python")
    if conda.is_file():
        return str(conda)
    return "python"


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
