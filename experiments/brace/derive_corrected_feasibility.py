#!/usr/bin/env python3
"""Derive corrected feasibility evidence offline from saved per-probe rows.

The scanner stores every individual probe under each result row's ``probes``
array together with ``probe_repeats``/``probe_passed_count``. A corrected
evidence file is recomputed from those raw probes under a fixed determinism
rule without any GPU work:

  python experiments/brace/derive_corrected_feasibility.py \
      --evidence experiments/brace/archive/seed_feasibility_reverify_20260808/lift_pot_supplemented_feasibility.json \
      --probe-repeats 2 --probe-success-rule all

The source evidence is never modified; the corrected file records the source
path and SHA256 under ``provenance.source_evidence`` and is itself validated
before it is written. Rows without per-probe data (e.g. single-probe runs)
cannot be recomputed offline and abort the derivation.
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
    PROBE_SUCCESS_RULES,
    build_feasibility_evidence,
    probe_passed,
    repo_code_commit,
    validate_feasibility_evidence,
)


def recompute_rows(rows: list[dict], *, probe_repeats: int, probe_success_rule: str) -> list[dict]:
    recomputed: list[dict] = []
    for row in rows:
        row_repeats = row.get("probe_repeats")
        if row_repeats != probe_repeats:
            raise ValueError(
                f"row seed {row.get('seed')} has probe_repeats={row_repeats}, "
                f"expected {probe_repeats}; cannot recompute offline"
            )
        passed_count = row.get("probe_passed_count")
        probes = row.get("probes")
        if isinstance(probes, list):
            recounted = sum(1 for probe in probes if probe.get("passed") is True)
            if passed_count is None:
                passed_count = recounted
            elif recounted != passed_count:
                raise ValueError(
                    f"row seed {row.get('seed')}: probes array ({recounted} passed) disagrees "
                    f"with probe_passed_count={passed_count}"
                )
        if passed_count is None:
            raise ValueError(
                f"row seed {row.get('seed')} has no per-probe data; cannot recompute offline"
            )
        new_row = json.loads(json.dumps(row))
        new_row["passed"] = probe_passed(passed_count, probe_repeats, probe_success_rule)
        if new_row["passed"]:
            new_row["error_type"] = None
            new_row["error_message"] = None
        else:
            failed = [probe for probe in probes if not probe.get("passed")] if isinstance(probes, list) else []
            if failed:
                new_row["error_type"] = failed[0].get("error_type")
                new_row["error_message"] = failed[0].get("error_message")
        recomputed.append(new_row)
    return recomputed


def build_corrected_evidence(
    source: dict,
    source_path: Path,
    *,
    probe_repeats: int,
    probe_success_rule: str,
    required_count: int,
) -> dict:
    rows = source.get("results", [])
    if not rows:
        raise ValueError(f"{source_path}: no result rows")
    recomputed = recompute_rows(rows, probe_repeats=probe_repeats, probe_success_rule=probe_success_rule)
    candidates = [int(row["seed"]) for row in recomputed]
    if len(candidates) != len(set(candidates)):
        raise ValueError(f"{source_path}: result rows contain duplicate seeds")

    source_provenance = source.get("provenance", {})
    provenance = {
        "code": source_provenance.get("code"),
        "software": source_provenance.get("software"),
        "task_config": source_provenance.get("task_config", source.get("task_config", "demo_clean")),
        "task_config_sha256": source_provenance.get("task_config_sha256"),
        "task_manifest_sha256": source_provenance.get("task_manifest_sha256"),
        "source_evidence": {
            "path": str(
                source_path.resolve().relative_to(REPO_ROOT.resolve())
                if source_path.resolve().is_relative_to(REPO_ROOT.resolve())
                else str(source_path)
            ),
            "sha256": file_sha256(source_path),
            "schema_version": source.get("schema_version"),
            "determinism": source.get("determinism"),
        },
    }

    supplement_source = source.get("supplement")
    supplement_block = None
    if supplement_source is not None:
        original_count = int(supplement_source.get("original_pool_candidate_count", 0))
        if original_count < 1 or original_count >= len(candidates):
            raise ValueError(f"{source_path}: invalid original_pool_candidate_count {original_count}")
        original_rows = recomputed[:original_count]
        supplement_rows = recomputed[original_count:]
        evidence = build_feasibility_evidence(
            task=str(source.get("task") or source_path.name),
            candidate_seeds=candidates,
            probe_results=recomputed,
            task_config=str(source.get("task_config", "demo_clean")),
            candidate_partition=str(source.get("candidate_partition", "rollout_train")),
            required_count=required_count,
            gpus=source.get("gpus"),
            shard_count=source.get("shard_count"),
            probe_repeats=probe_repeats,
            probe_success_rule=probe_success_rule,
        )
        base_seeds = {int(row["seed"]) for row in original_rows}
        used = [int(seed) for seed in evidence["expert_demo_seeds"] if int(seed) not in base_seeds]
        supplement_block = {
            "partition": str(supplement_source.get("partition", "expert_demo_supplement")),
            "count": len(supplement_rows),
            "original_pool_solvable": sum(1 for row in original_rows if row.get("passed") is True),
            "original_pool_candidate_count": original_count,
            "original_pool_task_status": str(supplement_source.get("original_pool_task_status")),
            "original_evidence_path": str(supplement_source.get("original_evidence_path")),
            "original_evidence_sha256": str(supplement_source.get("original_evidence_sha256")),
            "selection_rule": str(
                supplement_source.get("selection_rule", "original_pool_first_then_supplement_ascending")
            ),
            "supplement_solvable": sum(1 for row in supplement_rows if row.get("passed") is True),
            "supplement_shard_count": int(supplement_source.get("supplement_shard_count", 0)),
            "determinism": {"probe_repeats": probe_repeats, "probe_success_rule": probe_success_rule},
            "original_determinism": {
                "probe_repeats": int(supplement_source.get("original_determinism", {}).get("probe_repeats", probe_repeats)),
                "probe_success_rule": str(
                    supplement_source.get("original_determinism", {}).get("probe_success_rule", probe_success_rule)
                ),
            },
            "used": bool(used),
            "supplement_seeds_used": used,
        }
        evidence["supplement"] = supplement_block
    else:
        evidence = build_feasibility_evidence(
            task=str(source.get("task") or source_path.name),
            candidate_seeds=candidates,
            probe_results=recomputed,
            task_config=str(source.get("task_config", "demo_clean")),
            candidate_partition=str(source.get("candidate_partition", "rollout_train")),
            required_count=required_count,
            gpus=source.get("gpus"),
            shard_count=source.get("shard_count"),
            probe_repeats=probe_repeats,
            probe_success_rule=probe_success_rule,
        )
    if provenance.get("code") is None:
        provenance["code"] = {}
    evidence["provenance"] = provenance
    evidence["derivation"] = {
        "kind": "offline_recompute_from_saved_probes",
        "determinism": {"probe_repeats": probe_repeats, "probe_success_rule": probe_success_rule},
        "code": repo_code_commit(),
        "required_count": required_count,
    }
    if source.get("verify_label") is not None:
        evidence["verify_label"] = source["verify_label"]
    errors = validate_feasibility_evidence(evidence)
    if errors:
        raise ValueError(f"corrected evidence invalid: {errors}")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True, help="schema-v2 source evidence with saved probes")
    parser.add_argument("--output", type=Path, help="default: <stem>_corrected.json next to the source")
    parser.add_argument("--probe-repeats", type=int, default=2)
    parser.add_argument("--probe-success-rule", default="all")
    parser.add_argument("--required-count", type=int, default=DEFAULT_EXPERT_DEMO_COUNT)
    args = parser.parse_args()

    if args.probe_repeats < 1:
        raise SystemExit("--probe-repeats must be positive")
    if args.probe_success_rule not in PROBE_SUCCESS_RULES:
        raise SystemExit(f"--probe-success-rule must be one of {sorted(PROBE_SUCCESS_RULES)}")
    if not args.evidence.is_file():
        raise SystemExit(f"evidence file missing: {args.evidence}")

    source = read_json(args.evidence)
    if source.get("schema_version") != 2 or not source.get("task_status"):
        raise SystemExit(
            f"source evidence must be schema v2 with task_status: {args.evidence}"
        )
    output = args.output or args.evidence.with_name(args.evidence.stem + "_corrected.json")
    corrected = build_corrected_evidence(
        source,
        args.evidence.resolve(),
        probe_repeats=args.probe_repeats,
        probe_success_rule=args.probe_success_rule,
        required_count=args.required_count,
    )
    write_json_atomic(output, corrected)
    print(json.dumps({
        "task": corrected["task"],
        "task_status": corrected["task_status"],
        "passed": corrected["passed"],
        "solvable_count_in_candidates": corrected["solvable_count_in_candidates"],
        "supplement_used": bool(corrected.get("supplement", {}).get("used")),
        "source_evidence": corrected["provenance"]["source_evidence"]["path"],
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
