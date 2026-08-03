#!/usr/bin/env python3
"""Aggregate confirmatory preservation/adaptation results and evaluate H1 gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.confirmatory_common import seed_set_sha256
from experiments.brace.replay_audit import read_json, write_json_atomic


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_one_sided_sign_test(wins: int, n: int, *, alpha: float = 0.05) -> dict[str, Any]:
    if n <= 0:
        return {"wins": wins, "n": n, "p_value": None, "passed": False}
    p_value = sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n)
    return {
        "wins": wins,
        "n": n,
        "p_value": p_value,
        "passed": wins == n and p_value <= alpha,
    }


def zero_event_upper_bound_95(n: int) -> float | None:
    if n <= 0:
        return None
    return 1.0 - (0.05 ** (1.0 / n))


def load_eval_rows(eval_path: Path) -> list[dict[str, Any]]:
    return read_json(eval_path).get("rows", [])


def episode_outcomes_from_rows(rows: list[dict[str, Any]], split: str) -> dict[int, bool]:
    by_seed: dict[int, list[bool]] = {}
    for row in rows:
        if row.get("split") != split:
            continue
        by_seed.setdefault(int(row["env_seed"]), []).append(bool(row.get("success", False)))
    return {seed: all(values) for seed, values in by_seed.items()}


def mean_repeat_success_rate(rows: list[dict[str, Any]], split: str, env_seed: int) -> float:
    outcomes = [
        bool(row.get("success", False))
        for row in rows
        if row.get("split") == split and int(row["env_seed"]) == env_seed
    ]
    if not outcomes:
        raise KeyError(f"missing rows for split={split} env_seed={env_seed}")
    return sum(outcomes) / len(outcomes)


def preserved_on_cohort(rows: list[dict[str, Any]], split: str, env_seed: int) -> bool:
    return bool(episode_outcomes_from_rows(rows, split).get(env_seed, False))


def forgetting_rate(
    *,
    candidate_rows: list[dict[str, Any]],
    cohort_seeds: list[int],
    split: str,
) -> dict[str, Any]:
    candidate = episode_outcomes_from_rows(candidate_rows, split)
    missing = [seed for seed in cohort_seeds if seed not in candidate]
    if missing:
        raise ValueError(f"candidate is missing {split} outcomes for cohort seeds: {missing[:10]}")
    forgetting = [seed for seed in cohort_seeds if not candidate[seed]]
    return {
        "split": split,
        "denominator_source": "frozen_census_cohort",
        "paired_seeds": len(cohort_seeds),
        "forgetting_seeds": forgetting,
        "forgetting_count": len(forgetting),
        "forgetting_rate": len(forgetting) / len(cohort_seeds) if cohort_seeds else None,
    }


def validate_eval_payload(
    *,
    label: str,
    payload: dict[str, Any],
    protocol: dict[str, Any],
    cohort: dict[str, Any],
    expected_hard_seeds: list[int] | None = None,
) -> list[int]:
    eval_cfg = protocol["eval"]
    progress = payload.get("progress", {})
    expected_progress = {
        "complete": True,
        "id_repeats": int(eval_cfg["id_repeats"]),
        "train_repeats": int(eval_cfg["train_repeats"]),
        "hard_repeats": int(eval_cfg["hard_repeats"]),
        "extra_split_repeats": int(eval_cfg["extra_split_repeats"]),
        "policy_seed_offset": int(eval_cfg["policy_seed_offset"]),
    }
    mismatches = [
        f"progress.{key}={progress.get(key)!r}, expected {value!r}"
        for key, value in expected_progress.items()
        if progress.get(key) != value
    ]
    hard_seeds = [int(seed) for seed in payload.get("hard_seeds", [])]
    if expected_hard_seeds is not None and hard_seeds != expected_hard_seeds:
        mismatches.append("hard_seeds differ from base_original")
    expected: dict[str, tuple[list[int], int]] = {
        "id_heldout": (
            [int(seed) for seed in cohort["cohorts"]["id_heldout"]],
            int(eval_cfg["id_repeats"]),
        ),
        "train_seen": (
            [int(seed) for seed in cohort["cohorts"]["train_seen"]],
            int(eval_cfg["train_repeats"]),
        ),
        "hard_20": (hard_seeds, int(eval_cfg["hard_repeats"])),
    }
    for split in ("untouched_preservation", "anchor_train", "anchor_probe", "boundary"):
        expected[split] = (
            [int(seed) for seed in cohort["cohorts"].get(split, [])],
            int(eval_cfg["extra_split_repeats"]),
        )
    expected_keys = {
        (split, seed, repeat)
        for split, (seeds, repeats) in expected.items()
        for seed in seeds
        for repeat in range(repeats)
    }
    actual_keys: set[tuple[str, int, int]] = set()
    offset = int(eval_cfg["policy_seed_offset"])
    for row in payload.get("rows", []):
        key = (str(row.get("split")), int(row["env_seed"]), int(row.get("repeat", -1)))
        if key in actual_keys:
            mismatches.append(f"duplicate work item {key}")
            continue
        actual_keys.add(key)
        if int(row.get("policy_seed", -1)) != offset + key[2]:
            mismatches.append(f"wrong policy_seed for {key}")
    if actual_keys != expected_keys:
        mismatches.append(
            f"work item set differs: missing={len(expected_keys - actual_keys)}, "
            f"unexpected={len(actual_keys - expected_keys)}"
        )
    if mismatches:
        raise ValueError(f"invalid confirmatory eval artifact {label}: " + "; ".join(mismatches[:20]))
    return hard_seeds


def adaptation_delta(
    *,
    c0_rows: list[dict[str, Any]],
    c1_rows: list[dict[str, Any]],
    env_seeds: list[int],
    split: str,
) -> float:
    deltas = []
    for env_seed in env_seeds:
        c0_rate = mean_repeat_success_rate(c0_rows, split, env_seed)
        c1_rate = mean_repeat_success_rate(c1_rows, split, env_seed)
        deltas.append(c1_rate - c0_rate)
    return float(np.mean(deltas)) if deltas else float("nan")


def hierarchical_paired_bootstrap(
    *,
    values_by_training_seed: dict[int, float],
    env_values_by_training_seed: dict[int, dict[int, float]] | None,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    training_seeds = sorted(values_by_training_seed)
    if not training_seeds:
        return {"lower": None, "upper": None, "point": None}
    samples = []
    for _ in range(replicates):
        if env_values_by_training_seed:
            draw = []
            for _ in range(len(training_seeds)):
                ts = int(rng.choice(training_seeds))
                env_map = env_values_by_training_seed[ts]
                env_ids = sorted(env_map)
                env_draw = [env_map[int(rng.choice(env_ids))] for _ in range(len(env_ids))]
                draw.append(float(np.mean(env_draw)))
            samples.append(float(np.mean(draw)))
        else:
            draw = [values_by_training_seed[int(rng.choice(training_seeds))] for _ in range(len(training_seeds))]
            samples.append(float(np.mean(draw)))
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "point": float(np.mean([values_by_training_seed[ts] for ts in training_seeds])),
        "lower": float(np.quantile(samples, alpha)),
        "upper": float(np.quantile(samples, 1.0 - alpha)),
        "replicates": replicates,
    }


def enrollment_eligible(success_count: int, min_successes: int) -> bool:
    return success_count >= min_successes


def validate_census_not_used_for_confirmatory(eval_path: Path, census_eval_path: Path | None) -> None:
    if census_eval_path is not None and eval_path.resolve() == census_eval_path.resolve():
        raise ValueError("confirmatory eval must not reuse census eval artifact")


def aggregate_confirmatory_preservation(
    *,
    protocol: dict[str, Any],
    cohort: dict[str, Any],
    eval_artifacts: dict[str, Path],
    census_eval_path: Path | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    training_seeds = [int(seed) for seed in protocol["training_seeds"]]
    untouched = [int(seed) for seed in cohort["cohorts"]["untouched_preservation"]]
    id_seeds = [int(seed) for seed in cohort["cohorts"]["id_heldout"]]
    train_seeds = [int(seed) for seed in cohort["cohorts"].get("train_seen", [])]
    hard_seeds = sorted(
        {
            int(row["env_seed"])
            for row in load_eval_rows(eval_artifacts["base_original"])
            if row.get("split") == "hard_20"
        }
    )
    split_pres = protocol["preservation_gate"]["split"]
    split_adapt = protocol["adaptation_gate"]["split"]
    adapt_margin = float(protocol["adaptation_gate"]["noninferiority_margin_absolute"])

    base_rows = load_eval_rows(eval_artifacts["base_original"])
    validate_census_not_used_for_confirmatory(eval_artifacts["base_original"], census_eval_path)

    per_seed: dict[str, Any] = {}
    rel_pres_wins = 0
    adapt_wins = 0
    pres_deltas: dict[int, float] = {}
    adapt_deltas: dict[int, float] = {}
    pres_env_deltas: dict[int, dict[int, float]] = {}
    adapt_env_values: dict[int, dict[int, float]] = {}
    absolute_forgetting_total = 0
    secondary_by_seed: dict[str, Any] = {}

    for training_seed in training_seeds:
        c0_key = f"c0_s{training_seed}"
        c1_key = f"c1_s{training_seed}"
        c0_rows = load_eval_rows(eval_artifacts[c0_key])
        c1_rows = load_eval_rows(eval_artifacts[c1_key])
        validate_census_not_used_for_confirmatory(eval_artifacts[c0_key], census_eval_path)
        validate_census_not_used_for_confirmatory(eval_artifacts[c1_key], census_eval_path)

        c0_forget = forgetting_rate(candidate_rows=c0_rows, cohort_seeds=untouched, split=split_pres)
        c1_forget = forgetting_rate(candidate_rows=c1_rows, cohort_seeds=untouched, split=split_pres)
        delta_pres = (c1_forget["forgetting_rate"] or 0.0) - (c0_forget["forgetting_rate"] or 0.0)
        pres_deltas[training_seed] = delta_pres
        pres_env_deltas[training_seed] = {}
        for env_seed in untouched:
            c0_ok = preserved_on_cohort(c0_rows, split_pres, env_seed)
            c1_ok = preserved_on_cohort(c1_rows, split_pres, env_seed)
            pres_env_deltas[training_seed][env_seed] = float(not c1_ok) - float(not c0_ok)
        if delta_pres < 0:
            rel_pres_wins += 1

        absolute_forgetting_total += int(c1_forget["forgetting_count"] or 0)

        delta_adapt = adaptation_delta(c0_rows=c0_rows, c1_rows=c1_rows, env_seeds=id_seeds, split=split_adapt)
        adapt_deltas[training_seed] = delta_adapt
        adapt_env_values[training_seed] = {
            env_seed: mean_repeat_success_rate(c1_rows, split_adapt, env_seed)
            - mean_repeat_success_rate(c0_rows, split_adapt, env_seed)
            for env_seed in id_seeds
        }
        if delta_adapt > adapt_margin:
            adapt_wins += 1

        train_delta = adaptation_delta(
            c0_rows=c0_rows,
            c1_rows=c1_rows,
            env_seeds=train_seeds,
            split="train_seen",
        )
        hard_delta_c1_c0 = adaptation_delta(
            c0_rows=c0_rows,
            c1_rows=c1_rows,
            env_seeds=hard_seeds,
            split="hard_20",
        )
        hard_delta_c1_base = adaptation_delta(
            c0_rows=base_rows,
            c1_rows=c1_rows,
            env_seeds=hard_seeds,
            split="hard_20",
        )
        secondary_by_seed[str(training_seed)] = {
            "train_seen_c1_minus_c0": train_delta,
            "hard_20_c1_minus_c0": hard_delta_c1_c0,
            "hard_20_c1_minus_base": hard_delta_c1_base,
            "gate_eligible": False,
        }

        per_seed[str(training_seed)] = {
            "c0_forgetting": c0_forget,
            "c1_forgetting": c1_forget,
            "delta_preservation_forgetting_rate": delta_pres,
            "delta_adaptation_id": delta_adapt,
        }

    n_untouched = len(untouched)
    if n_untouched < int(protocol["preservation_cohorts"]["min_untouched_base_solved"]):
        raise ValueError(f"frozen untouched cohort is too small: {n_untouched}")
    pres_sign = exact_one_sided_sign_test(rel_pres_wins, len(training_seeds))
    adapt_sign = exact_one_sided_sign_test(adapt_wins, len(training_seeds))
    ci_cfg = protocol["training_seed_inference"]["effect_ci"]
    pres_ci = hierarchical_paired_bootstrap(
        values_by_training_seed={ts: -pres_deltas[ts] for ts in training_seeds},
        env_values_by_training_seed={ts: {k: -v for k, v in pres_env_deltas[ts].items()} for ts in training_seeds},
        replicates=int(ci_cfg["replicates"]),
        seed=int(ci_cfg["bootstrap_seed"]),
        confidence_level=float(ci_cfg["confidence_level"]),
    )
    adapt_ci = hierarchical_paired_bootstrap(
        values_by_training_seed=adapt_deltas,
        env_values_by_training_seed=adapt_env_values,
        replicates=int(ci_cfg["replicates"]),
        seed=int(ci_cfg["bootstrap_seed"]) + 1,
        confidence_level=float(ci_cfg["confidence_level"]),
    )

    zero_upper = zero_event_upper_bound_95(n_untouched)
    zero_target = float(protocol["preservation_gate"]["zero_event_upper_bound_target"])
    gates = {
        "preservation_absolute_gate": {
            "passed": absolute_forgetting_total == 0 and zero_upper is not None and zero_upper <= zero_target,
            "forgetting_events": absolute_forgetting_total,
            "cohort_n": n_untouched,
            "training_seed_exposures": n_untouched * len(training_seeds),
            "zero_event_upper_bound_95_per_training_seed": zero_upper,
            "target": zero_target,
        },
        "preservation_relative_gate": {
            "passed": rel_pres_wins == len(training_seeds),
            "directional_wins": rel_pres_wins,
            "required_wins": len(training_seeds),
        },
        "preservation_sign_test": pres_sign,
        "preservation_bootstrap_ci": {
            "passed": pres_ci["lower"] is not None and pres_ci["lower"] > 0,
            **pres_ci,
        },
        "adaptation_noninferiority_gate": {
            "passed": adapt_wins == len(training_seeds),
            "directional_wins": adapt_wins,
            "required_wins": len(training_seeds),
            "margin_absolute": adapt_margin,
        },
        "adaptation_sign_test": adapt_sign,
        "adaptation_bootstrap_ci": {
            "passed": adapt_ci["lower"] is not None and adapt_ci["lower"] > adapt_margin,
            **adapt_ci,
        },
        "artifact_provenance_complete": {
            "passed": bool(provenance and provenance.get("complete", False)),
            "details": provenance or {},
        },
    }

    conjunction = protocol.get("h1_conjunction", [])
    failed = [name for name in conjunction if not gates.get(name, {}).get("passed", False)]
    h1_status = "passed" if not failed else "failed"

    return {
        "schema_version": 1,
        "stage": "confirmatory_preservation_aggregate",
        "h1_status": h1_status,
        "failed_gates": failed,
        "per_training_seed": per_seed,
        "base_fresh_preservation_diagnostic": forgetting_rate(
            candidate_rows=base_rows,
            cohort_seeds=untouched,
            split=split_pres,
        ),
        "secondary_adaptation": {
            "note": "Frozen reporting only; excluded from H1 conjunction and checkpoint selection.",
            "per_training_seed": secondary_by_seed,
            "mean_train_seen_c1_minus_c0": float(
                np.mean([row["train_seen_c1_minus_c0"] for row in secondary_by_seed.values()])
            ),
            "mean_hard_20_c1_minus_c0": float(
                np.mean([row["hard_20_c1_minus_c0"] for row in secondary_by_seed.values()])
            ),
            "mean_hard_20_c1_minus_base": float(
                np.mean([row["hard_20_c1_minus_base"] for row in secondary_by_seed.values()])
            ),
        },
        "gates": gates,
        "cohort_untouched_n": n_untouched,
        "protocol_revision": protocol.get("protocol_revision"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=BRACE_DIR / "screen_protocol.v1.4.1.confirmatory_preservation.json")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--census-eval", type=Path, required=True, help="Census eval path; confirmatory eval must differ")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = read_json(args.protocol.resolve())
    cohort = read_json(args.cohort.resolve())
    protocol_sha256 = file_sha256(args.protocol.resolve())
    cohort_sha256 = file_sha256(args.cohort.resolve())
    if protocol.get("status") != "frozen" or bool(protocol.get("exploratory", True)):
        raise SystemExit("H1 report requires a frozen non-exploratory protocol")
    if not cohort.get("frozen") or not cohort.get("meets_min_untouched"):
        raise SystemExit("H1 report requires a frozen eligible cohort")
    if cohort.get("protocol_sha256") != protocol_sha256:
        raise SystemExit("cohort protocol SHA mismatch")
    census_eval_path = args.census_eval.resolve()
    if not census_eval_path.is_file():
        raise SystemExit(f"missing census eval artifact: {census_eval_path}")
    if Path(str(cohort.get("census_eval_path", ""))).resolve() != census_eval_path:
        raise SystemExit("--census-eval does not match the cohort provenance")
    if cohort.get("input_sha256", {}).get("census_eval") != file_sha256(census_eval_path):
        raise SystemExit("census eval SHA mismatch")
    frozen_seed_sha = seed_set_sha256(
        [int(seed) for seed in cohort["cohorts"]["id_heldout"]]
        + [int(seed) for seed in cohort["cohorts"]["train_seen"]]
    )
    if cohort.get("seed_set_sha256") != frozen_seed_sha:
        raise SystemExit("cohort seed_set_sha256 mismatch")
    eval_dir = args.eval_dir.resolve()
    task = cohort.get("task", "place_container_plate")
    eval_artifacts = {"base_original": eval_dir / task / "confirm_base_original.json"}
    for training_seed in protocol["training_seeds"]:
        eval_artifacts[f"c0_s{training_seed}"] = eval_dir / task / f"confirm_c0_s{training_seed}.json"
        eval_artifacts[f"c1_s{training_seed}"] = eval_dir / task / f"confirm_c1_s{training_seed}.json"
    missing = [key for key, path in eval_artifacts.items() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing confirmatory eval artifacts: {missing}")

    partials: dict[str, dict[str, Any]] = {}
    expected_labels = set(eval_artifacts)
    for label, eval_path in eval_artifacts.items():
        partial_path = eval_dir / "partials" / f"{label}.json"
        if not partial_path.is_file():
            raise SystemExit(f"missing confirmatory partial: {partial_path}")
        partial = read_json(partial_path)
        partials[label] = partial
        if partial.get("label") != label:
            raise SystemExit(f"partial label mismatch: {partial_path}")
        if Path(str(partial.get("eval_output", ""))).resolve() != eval_path.resolve():
            raise SystemExit(f"partial eval_output mismatch: {partial_path}")
        if partial.get("eval_sha256") != file_sha256(eval_path):
            raise SystemExit(f"partial eval SHA mismatch: {partial_path}")
        if partial.get("protocol_sha256") != protocol_sha256 or partial.get("cohort_sha256") != cohort_sha256:
            raise SystemExit(f"partial protocol/cohort provenance mismatch: {partial_path}")
        checkpoint = Path(str(partial.get("checkpoint_path", "")))
        if not checkpoint.is_file() or partial.get("checkpoint_sha256") != file_sha256(checkpoint):
            raise SystemExit(f"partial checkpoint provenance mismatch: {partial_path}")
        if not partial.get("checkpoint_audit", {}).get("deploys_ema"):
            raise SystemExit(f"partial checkpoint does not deploy EMA: {partial_path}")
        if label != "base_original":
            expected_seed = int(label.rsplit("_s", 1)[1])
            if int(partial.get("training_seed", -1)) != expected_seed:
                raise SystemExit(f"partial training seed mismatch: {partial_path}")
    if set(partials) != expected_labels:
        raise SystemExit("confirmatory partial label set mismatch")

    base_payload = read_json(eval_artifacts["base_original"])
    hard_seeds = validate_eval_payload(
        label="base_original", payload=base_payload, protocol=protocol, cohort=cohort
    )
    for label, path in eval_artifacts.items():
        if label == "base_original":
            continue
        validate_eval_payload(
            label=label,
            payload=read_json(path),
            protocol=protocol,
            cohort=cohort,
            expected_hard_seeds=hard_seeds,
        )

    provenance = {
        "complete": True,
        "protocol_sha256": protocol_sha256,
        "cohort_sha256": cohort_sha256,
        "census_eval_sha256": file_sha256(census_eval_path),
        "eval_sha256": {key: file_sha256(path) for key, path in eval_artifacts.items()},
        "partial_sha256": {
            key: file_sha256(eval_dir / "partials" / f"{key}.json") for key in eval_artifacts
        },
        "checkpoint_sha256": {key: partials[key]["checkpoint_sha256"] for key in eval_artifacts},
    }
    summary = aggregate_confirmatory_preservation(
        protocol=protocol,
        cohort=cohort,
        eval_artifacts=eval_artifacts,
        census_eval_path=census_eval_path,
        provenance=provenance,
    )
    summary["operational_probe_diagnostic"] = {
        label: partials[label].get("operational_probe_gate")
        for label in sorted(partials)
        if label.startswith("c1_s")
    }
    summary["provenance"] = provenance
    write_json_atomic(args.output.resolve(), summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["h1_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
