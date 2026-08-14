#!/usr/bin/env python3
"""Write frozen expert_demo seeds into data/<task>/demo_clean/seed.txt.

Fail-closed gates (Line B default):
  - manifest status == frozen
  - feasibility.passed == true
  - feasibility.evidence_sha256 matches the evidence file on disk
  - manifest SHA256 sidecar is valid
  - expert_demo cohort is exactly `count` seeds, subset of
    rollout_train ∪ expert_demo_supplement

Setting BRACE_ALLOW_PROVISIONAL_ASSETS=1 skips the frozen gate for exploratory
assets only: seeds are written under data/<task>/demo_clean_exploratory/ and
the cohort provenance is marked exploratory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.multitask_protocol import file_sha256, sha256_sidecar_valid
from experiments.brace.seed_feasibility import (
    DEFAULT_EXPERT_DEMO_COUNT,
    TASK_STATUS_PASSED,
    validate_feasibility_evidence,
)

PROVISIONAL_ASSETS_ENV = "BRACE_ALLOW_PROVISIONAL_ASSETS"
EXPLORATORY_DIR_SUFFIX = "_exploratory"


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def resolve_expert_demo_seeds(manifest: dict, *, count: int) -> list[int]:
    selection = manifest.get("expert_demo_selection", {})
    cohorts = manifest.get("cohorts", {})
    seeds = cohorts.get("expert_demo", [])
    if not seeds and "expert_demo" in manifest.get("partitions", {}):
        raise SystemExit(
            f"{manifest.get('task', '<task>')}: partitions.expert_demo is deprecated; "
            "move the cohort to cohorts.expert_demo"
        )
    if not seeds:
        raise SystemExit(
            f"{manifest.get('task', '<task>')}: cohorts.expert_demo is empty; "
            "run scan_seed_feasibility.py --update-manifest first"
        )
    if len(seeds) != count:
        raise SystemExit(
            f"expected {count} expert_demo seeds, found {len(seeds)} "
            f"(rule={selection.get('rule')})"
        )
    rollout_train = set(manifest.get("partitions", {}).get("rollout_train", []))
    supplement = set(manifest.get("partitions", {}).get("expert_demo_supplement", []))
    if not set(seeds).issubset(rollout_train | supplement):
        raise SystemExit(
            "expert_demo seeds must remain a subset of rollout_train ∪ expert_demo_supplement"
        )
    return [int(seed) for seed in seeds]


def check_manifest_frozen(manifest: dict, manifest_path: Path) -> None:
    errors: list[str] = []
    task = manifest.get("task", manifest_path.stem)
    if manifest.get("status") != "frozen":
        errors.append(f"{task}: manifest status is {manifest.get('status')!r}, expected 'frozen'")
    feasibility = manifest.get("feasibility", {})
    if feasibility.get("passed") is not True:
        errors.append(f"{task}: feasibility.passed is not true")
    if not sha256_sidecar_valid(manifest_path):
        errors.append(f"{task}: manifest SHA256 sidecar {manifest_path}.sha256 is missing or invalid")
    evidence_sha = feasibility.get("evidence_sha256")
    evidence_path = feasibility.get("evidence_path")
    if not evidence_sha:
        errors.append(f"{task}: feasibility.evidence_sha256 is missing")
    if not evidence_path:
        errors.append(f"{task}: feasibility.evidence_path is missing")
    if evidence_sha and evidence_path:
        resolved = Path(evidence_path) if Path(evidence_path).is_absolute() else REPO_ROOT / evidence_path
        if not resolved.is_file():
            errors.append(f"{task}: feasibility evidence file missing: {resolved}")
        elif file_sha256(resolved) != evidence_sha:
            errors.append(f"{task}: feasibility evidence SHA256 mismatch on {resolved}")
        else:
            evidence = json.loads(resolved.read_text(encoding="utf-8"))
            if evidence.get("task_status") != TASK_STATUS_PASSED:
                errors.append(f"{task}: feasibility evidence task_status is not 'passed'")
            evidence_errors = validate_feasibility_evidence(evidence)
            if evidence_errors:
                errors.append(f"{task}: feasibility evidence invalid: {evidence_errors}")
    if errors:
        raise SystemExit(
            "\n".join(errors)
            + f"\nSet {PROVISIONAL_ASSETS_ENV}=1 to allow exploratory assets (marked exploratory)."
        )


def cohort_provenance(
    manifest: dict,
    manifest_path: Path,
    seeds: list[int],
    *,
    exploratory: bool,
    data_dir: Path,
) -> dict:
    evidence_path = manifest.get("feasibility", {}).get("evidence_path")
    return {
        "schema_version": 1,
        "task": manifest.get("task", manifest_path.stem),
        "exploratory": exploratory,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expert_demo_seeds": seeds,
        "expert_demo_count": len(seeds),
        "selection_rule": manifest.get("expert_demo_selection", {}).get("rule"),
        "manifest_sha256": file_sha256(manifest_path),
        "evidence_path": evidence_path,
        "evidence_sha256": manifest.get("feasibility", {}).get("evidence_sha256"),
        "task_config_sha256": file_sha256(REPO_ROOT / "task_config" / "demo_clean.yml"),
        "code_commit": git_commit(),
        "seed_txt_path": str((data_dir / "seed.txt").resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--count", type=int, default=DEFAULT_EXPERT_DEMO_COUNT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=BRACE_DIR / "seeds" / "multitask_v1",
    )
    args = parser.parse_args()

    manifest_path = args.manifest / f"{args.task}.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    exploratory = os.environ.get(PROVISIONAL_ASSETS_ENV, "0") == "1"
    if not exploratory:
        check_manifest_frozen(payload, manifest_path)
    else:
        print(
            f"WARNING: {PROVISIONAL_ASSETS_ENV}=1: skipping frozen gate; "
            "output is exploratory only and must not feed Line B"
        )

    seeds = resolve_expert_demo_seeds(payload, count=args.count)

    base_dir = REPO_ROOT / "data" / args.task
    if exploratory:
        save_dir = base_dir / f"demo_clean{EXPLORATORY_DIR_SUFFIX}"
    else:
        save_dir = base_dir / "demo_clean"
    save_dir.mkdir(parents=True, exist_ok=True)
    seed_path = save_dir / "seed.txt"
    seed_path.write_text(" ".join(str(seed) for seed in seeds) + "\n", encoding="utf-8")
    provenance = cohort_provenance(payload, manifest_path, seeds, exploratory=exploratory, data_dir=save_dir)
    provenance_path = save_dir / "cohort_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {len(seeds)} expert_demo seeds to {seed_path} "
        f"({seeds[0]}..{seeds[-1]}) rule={payload.get('expert_demo_selection', {}).get('rule')} "
        f"exploratory={exploratory}"
    )
    print(f"wrote cohort provenance to {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
