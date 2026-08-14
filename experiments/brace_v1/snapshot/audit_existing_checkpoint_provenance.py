#!/usr/bin/env python3
"""Audit whether existing demo data / zarr / checkpoint provenance is provable.

The existence of a checkpoint or zarr is not sufficient evidence that it was
trained on the currently frozen cohort. For each task this audits:

  - seed manifest frozen status and SHA256 sidecar validity;
  - cohorts.expert_demo seed list;
  - data/<task>/demo_clean/cohort_provenance.json consistency with the manifest;
  - presence of demo HDF5, zarr and checkpoint.

Verdict per task:
  reuse-ok      assets exist and provenance matches the frozen manifest
  must-rebuild  assets exist but provenance cannot be proven
  no-assets     nothing to audit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.multitask_protocol import file_sha256, sha256_sidecar_valid

DEFAULT_TASKS = ["click_alarmclock", "move_can_pot", "put_object_cabinet"]
REQUIRED_EPISODES = 50


def audit_task(task: str, *, required_episodes: int) -> dict:
    manifest_path = BRACE_DIR / "seeds" / "multitask_v1" / f"{task}.json"
    row: dict = {
        "task": task,
        "manifest_exists": manifest_path.is_file(),
        "manifest_sidecar_valid": sha256_sidecar_valid(manifest_path) if manifest_path.is_file() else False,
        "manifest_frozen": False,
        "cohort_count": 0,
        "provenance_match": None,
        "demo_episode_count": 0,
        "zarr_exists": False,
        "checkpoint_exists": False,
        "verdict": "no-assets",
        "details": [],
    }
    if not manifest_path.is_file():
        row["details"].append("missing seed manifest")
        return row

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row["manifest_frozen"] = manifest.get("status") == "frozen" and manifest.get("feasibility", {}).get("passed") is True
    cohort = manifest.get("cohorts", {}).get("expert_demo", [])
    row["cohort_count"] = len(cohort)
    if not row["manifest_frozen"]:
        row["details"].append(
            f"manifest status={manifest.get('status')!r} feasibility.passed={manifest.get('feasibility', {}).get('passed')}"
        )

    demo_dir = REPO_ROOT / "data" / task / "demo_clean"
    seed_txt = demo_dir / "seed.txt"
    provenance_path = demo_dir / "cohort_provenance.json"
    expected_seeds = " ".join(str(int(seed)) for seed in cohort)
    if not seed_txt.is_file():
        row["details"].append("no demo_clean/seed.txt")
    elif not provenance_path.is_file():
        row["details"].append("demo_clean/cohort_provenance.json missing")
    else:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        checks = [
            provenance.get("manifest_sha256") == file_sha256(manifest_path),
            provenance.get("evidence_sha256") == manifest.get("feasibility", {}).get("evidence_sha256"),
            seed_txt.read_text(encoding="utf-8").strip() == expected_seeds,
        ]
        row["provenance_match"] = all(checks)
        if not row["provenance_match"]:
            row["details"].append("cohort provenance does not match the manifest")
    data_dir = demo_dir / "data"
    if data_dir.is_dir():
        row["demo_episode_count"] = sum(1 for p in data_dir.iterdir() if p.name.startswith("episode") and p.suffix == ".hdf5")

    zarr = REPO_ROOT / "policy" / "DP" / "data" / f"{task}-demo_clean-{required_episodes}.zarr"
    ckpt = REPO_ROOT / "policy" / "DP" / "checkpoints" / f"{task}-demo_clean-{required_episodes}-0" / "600.ckpt"
    row["zarr_exists"] = zarr.is_dir()
    row["checkpoint_exists"] = ckpt.is_file()

    if not (row["zarr_exists"] or row["checkpoint_exists"]):
        row["verdict"] = "no-assets"
    elif row["manifest_frozen"] and row["provenance_match"] and row["cohort_count"] == required_episodes:
        row["verdict"] = "reuse-ok"
    else:
        row["verdict"] = "must-rebuild"
        row["details"].append("existing zarr/checkpoint provenance is unproven; rebuild required")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--required-episodes", type=int, default=REQUIRED_EPISODES)
    args = parser.parse_args()
    rows = [audit_task(task, required_episodes=args.required_episodes) for task in args.tasks]
    print(json.dumps({"schema_version": 1, "tasks": rows}, indent=2))
    rebuild = [row["task"] for row in rows if row["verdict"] == "must-rebuild"]
    if rebuild:
        print(f"must-rebuild: {rebuild}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
