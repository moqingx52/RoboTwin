#!/usr/bin/env python3
"""Atomically freeze a seed manifest from validated feasibility evidence.

All held-out seed manifests must be frozen under one uniform probe
determinism criterion before Line B. This command performs every check and
write in one step:

  - evidence schema v2 and validator-clean;
  - evidence.task_status == passed and task matches;
  - determinism == --required-probe-repeats/--required-probe-success-rule;
  - result seed set matches the manifest pool exactly in manifest order
    (rollout_train, or rollout_train + expert_demo_supplement when the
    supplement amendment is used);
  - supplement amendment file is frozen and consistent with the manifest;
  - writes cohorts.expert_demo, feasibility.passed=true, evidence path+SHA,
    status=frozen, refreshes the manifest SHA256 sidecar, then re-reads and
    re-validates everything.

The evidence file must already live at its final archived location
(e.g. experiments/brace/archive/<run>/<task>_feasibility_corrected.json).

Usage:
    python experiments/brace/freeze_multitask_manifest.py \
        --task lift_pot \
        --evidence experiments/brace/archive/seed_feasibility_reverify_20260808/lift_pot_supplemented_feasibility_corrected.json
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

from experiments.brace.multitask_protocol import file_sha256
from experiments.brace.replay_audit import read_json, write_json_atomic
from experiments.brace.seed_feasibility import (
    DEFAULT_EXPERT_DEMO_COUNT,
    TASK_STATUS_PASSED,
    validate_feasibility_evidence,
    validate_seed_manifest_layout,
)

AMENDMENT_PATH = BRACE_DIR / "multitask_amendment.v1.1.json"


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def freeze_manifest(
    task: str,
    evidence_path: Path,
    manifest_path: Path,
    *,
    required_repeats: int,
    required_rule: str,
    required_count: int,
) -> dict:
    errors: list[str] = []
    manifest = read_json(manifest_path)
    errors.extend(validate_seed_manifest_layout(manifest))
    if manifest.get("task") != task:
        errors.append(f"manifest task {manifest.get('task')} != {task}")

    evidence = read_json(evidence_path)
    if evidence.get("task") != task:
        errors.append(f"evidence task {evidence.get('task')} != {task}")
    errors.extend(validate_feasibility_evidence(evidence))
    if evidence.get("task_status") != TASK_STATUS_PASSED:
        errors.append("evidence task_status must be 'passed' to freeze")

    determinism = evidence.get("determinism", {})
    if (
        determinism.get("probe_repeats") != required_repeats
        or determinism.get("probe_success_rule") != required_rule
    ):
        errors.append(
            f"evidence determinism {determinism} must be "
            f"{required_repeats}/{required_rule}"
        )

    supplement = evidence.get("supplement")
    pool = list(manifest.get("partitions", {}).get("rollout_train", []))
    if supplement is not None and supplement.get("used"):
        pool += list(manifest.get("partitions", {}).get("expert_demo_supplement", []))
        amendment = read_json(AMENDMENT_PATH) if AMENDMENT_PATH.is_file() else {}
        if amendment.get("status") != "frozen":
            errors.append("supplement evidence requires a frozen brace.multitask.v1.1 amendment")
        if supplement.get("partition") != "expert_demo_supplement":
            errors.append(f"supplement.partition must be expert_demo_supplement, got {supplement.get('partition')}")
    result_seeds = [int(row["seed"]) for row in evidence.get("results", [])]
    if result_seeds != [int(seed) for seed in pool]:
        errors.append(
            "evidence result seeds do not match the manifest pool exactly in manifest order "
            "(rollout_train or rollout_train + expert_demo_supplement)"
        )

    selected = [int(seed) for seed in evidence.get("expert_demo_seeds", [])]
    if len(selected) != required_count or len(set(selected)) != len(selected):
        errors.append(f"evidence must select exactly {required_count} unique seeds")
    if not set(selected).issubset(set(pool)):
        errors.append("expert_demo seeds must be a subset of the manifest pool")

    if errors:
        capped = errors[:15] + (["... and %d more" % (len(errors) - 15)] if len(errors) > 15 else [])
        raise ValueError("; ".join(capped))

    evidence_sha = file_sha256(evidence_path)
    rel_evidence = _repo_relative(evidence_path)
    updated = json.loads(json.dumps(manifest))
    updated.setdefault("cohorts", {})["expert_demo"] = selected
    updated["feasibility"].update(
        {
            "criterion": evidence.get("criterion"),
            "learning_policy_performance_consulted": evidence.get("learning_policy_performance_consulted", False),
            "passed": True,
            "evidence_path": rel_evidence,
            "evidence_sha256": evidence_sha,
        }
    )
    updated["expert_demo_selection"] = {
        "source_partition": evidence.get("candidate_partition", "rollout_train"),
        "rule": evidence.get("selection_rule", "first_n_solvable_in_manifest_order"),
        "required_count": required_count,
        "supplement_used": bool(supplement and supplement.get("used")),
        "evidence_path": rel_evidence,
        "evidence_sha256": evidence_sha,
    }
    updated["status"] = "frozen"
    write_json_atomic(manifest_path, updated)
    sidecar = Path(str(manifest_path) + ".sha256")
    sidecar.write_text(f"{file_sha256(manifest_path)}  {manifest_path.name}\n", encoding="utf-8")

    # Re-read and re-validate the frozen manifest.
    frozen = read_json(manifest_path)
    recheck: list[str] = []
    recheck.extend(validate_seed_manifest_layout(frozen))
    if frozen.get("status") != "frozen":
        recheck.append("status is not frozen after write")
    if frozen.get("feasibility", {}).get("passed") is not True:
        recheck.append("feasibility.passed is not true after write")
    if frozen.get("feasibility", {}).get("evidence_sha256") != evidence_sha:
        recheck.append("evidence_sha256 mismatch after write")
    if [int(seed) for seed in frozen.get("cohorts", {}).get("expert_demo", [])] != selected:
        recheck.append("cohort mismatch after write")
    sidecar_text = Path(str(manifest_path) + ".sha256").read_text(encoding="utf-8").strip().split()
    if not sidecar_text or sidecar_text[0] != file_sha256(manifest_path):
        recheck.append("manifest SHA256 sidecar invalid after write")
    if recheck:
        raise ValueError("freeze verification failed: " + "; ".join(recheck))
    return {
        "task": task,
        "status": "frozen",
        "cohort_count": len(selected),
        "supplement_used": bool(supplement and supplement.get("used")),
        "evidence_path": rel_evidence,
        "evidence_sha256": evidence_sha,
        "manifest_path": _repo_relative(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=BRACE_DIR / "seeds" / "multitask_v1" / "PLACEHOLDER.json",
    )
    parser.add_argument("--manifest-dir", type=Path, default=BRACE_DIR / "seeds" / "multitask_v1")
    parser.add_argument("--required-probe-repeats", type=int, default=2)
    parser.add_argument("--required-probe-success-rule", default="all")
    parser.add_argument("--required-count", type=int, default=DEFAULT_EXPERT_DEMO_COUNT)
    args = parser.parse_args()

    manifest_path = args.manifest
    if manifest_path.name == "PLACEHOLDER.json":
        manifest_path = args.manifest_dir / f"{args.task}.json"
    if not manifest_path.is_file():
        raise SystemExit(f"manifest file missing: {manifest_path}")
    if not args.evidence.is_file():
        raise SystemExit(f"evidence file missing: {args.evidence}")

    result = freeze_manifest(
        args.task,
        args.evidence.resolve(),
        manifest_path,
        required_repeats=args.required_probe_repeats,
        required_rule=args.required_probe_success_rule,
        required_count=args.required_count,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
