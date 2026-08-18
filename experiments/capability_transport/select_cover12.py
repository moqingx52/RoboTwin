#!/usr/bin/env python
"""Deterministic T2 arm selection from the frozen audit output.

Step 4 of the freeze ordering in protocol.t2_expert_audit.v1.json: a PURE
FUNCTION of (frozen cells file, frozen probe output, frozen T1c D_H
manifest). No simulator, no RNG beyond the frozen RANDOM_ARM_SEED constant,
no judgment calls — re-running always yields byte-identical selections.

Selections produced:
  Expert-Cover-12   one seed per capability cell: walk the cell's frozen
                    ranking, take the first expert-feasible seed not already
                    selected; a cell with no feasible member falls to its
                    frozen fallback-neighbor list. Cells are processed in the
                    frozen cell-key order of the cells file.
  Expert-Random-12  12 seeds sampled uniformly without replacement from the
                    sorted feasible set with random.Random(RANDOM_ARM_SEED).
  Self-Diverse-12   12 T1c D_H self-trajectories maximizing seed diversity:
                    round-robin over distinct successful seeds in ascending
                    seed order, within a seed by attempt_index ascending.
  Dose points       Q=6  = the first 6 slots of Expert-Cover-12 in cell order;
                    Q=20 = Expert-Cover-12 + the next 8 feasible seeds from
                    the frozen global ranking (DROPPED if < 20 feasible).

ESS at selection (realized, not idealized): adding one expert trajectory to
a seed that already has c_i T1c self successes gives
    n_eff = (41 + Q)^2 / (267 + Q + 2 * sum(c_i over selected seeds))
which reduces to the frozen bound (41+Q)^2/(267+Q) only when every selected
seed has c_i = 0. The arm is labeled an ESS rescue only if realized
n_eff >= 10.

Usage (inside the cloud container, from /workspace/RoboTwin):
    python experiments/capability_transport/select_cover12.py \
        --task place_container_plate \
        --t1c-run-dir <T1c run dir with manifest_D_H.jsonl>
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

CT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CT_DIR))

from common import (  # noqa: E402
    REPO_ROOT,
    iter_jsonl,
    read_json,
    write_json_atomic,
)

PROTOCOL_FILE = CT_DIR / "protocol.t2_expert_audit.v1.json"
RANDOM_ARM_SEED = 20260818  # frozen; used ONLY for Expert-Random-12
COVER_SIZE = 12
N_EFF_MIN = 10
T1C_DH_SUM = 41
T1C_DH_SUMSQ = 267


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_companion_sha256(path: Path) -> str:
    companion = path.with_suffix(path.suffix + ".sha256")
    if not companion.exists():
        raise SystemExit(f"missing sha256 companion for frozen input: {companion}")
    recorded = companion.read_text().split()[0]
    actual = sha256_file(path)
    if recorded != actual:
        raise SystemExit(f"sha256 mismatch for {path}: recorded {recorded}, actual {actual}")
    return actual


def realized_ess(q: int, selected_counts) -> float:
    return (T1C_DH_SUM + q) ** 2 / (T1C_DH_SUMSQ + q + 2 * sum(selected_counts))


def select_cover(cells_payload: dict, feasible: set) -> tuple[list, list]:
    """Frozen Cover selection: (selected seeds in slot order, per-slot log)."""
    selected = []
    slots = []
    cell_keys = list(cells_payload["cells"].keys())
    for key in cell_keys:
        cell = cells_payload["cells"][key]
        pick = None
        source_cell = key
        for s in cell["ranking"]:
            if s in feasible and s not in selected:
                pick = s
                break
        if pick is None:
            for nkey in cell["fallback_neighbors"]:
                for s in cells_payload["cells"][nkey]["ranking"]:
                    if s in feasible and s not in selected:
                        pick = s
                        source_cell = nkey
                        break
                if pick is not None:
                    break
        slots.append({
            "slot_cell": key,
            "seed": pick,
            "taken_from_cell": source_cell if pick is not None else None,
            "via_fallback": pick is not None and source_cell != key,
        })
        if pick is not None:
            selected.append(pick)
    return selected, slots


def select_self_diverse(dh_manifest: Path, states_per_seed: dict, k: int) -> list:
    """Frozen Self-Diverse rule over T1c D_H success rows (see module doc)."""
    by_seed = {}
    for row in iter_jsonl(dh_manifest):
        if row.get("group", "D_H") != "D_H":
            continue
        if row.get("success"):
            by_seed.setdefault(int(row["env_seed"]), []).append(row)
    # cross-check against the frozen T1c group states
    expect = {int(s): v["successes"] for s, v in states_per_seed.items() if v["successes"] > 0}
    got = {s: len(rows) for s, rows in by_seed.items()}
    if got != expect:
        raise SystemExit(
            f"D_H manifest success counts {got} do not match frozen "
            f"t1c_group_states {expect}; wrong manifest?"
        )
    total = sum(got.values())
    if total < k:
        raise SystemExit(f"only {total} D_H self successes; cannot build Self-Diverse-{k}")
    for rows in by_seed.values():
        rows.sort(key=lambda r: int(r["attempt_index"]))
    picks = []
    rnd = 0
    seeds_sorted = sorted(by_seed)
    while len(picks) < k:
        took_any = False
        for s in seeds_sorted:
            if len(picks) >= k:
                break
            if rnd < len(by_seed[s]):
                row = by_seed[s][rnd]
                picks.append({
                    "env_seed": s,
                    "attempt_index": int(row["attempt_index"]),
                    "hdf5_path": row.get("hdf5_path"),
                })
                took_any = True
        if not took_any:
            raise SystemExit("Self-Diverse round-robin exhausted before k picks (impossible if counts checked)")
        rnd += 1
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="place_container_plate")
    parser.add_argument("--t1c-run-dir", type=Path, required=True,
                        help="T1c run dir containing the merged manifest_D_H.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    protocol_sha = verify_companion_sha256(PROTOCOL_FILE)
    protocol = read_json(PROTOCOL_FILE)
    rules = protocol["collector_rules"]
    if rules["task"] != args.task:
        raise SystemExit("task does not match the frozen audit protocol")

    cells_file = REPO_ROOT / rules["cells_file"]
    cells_sha = verify_companion_sha256(cells_file)
    cells_payload = read_json(cells_file)

    audit_file = REPO_ROOT / rules["output_file"]
    if not audit_file.exists():
        raise SystemExit(f"frozen audit output missing: {audit_file}; run the probe first")
    audit_sha = verify_companion_sha256(audit_file)
    audit = read_json(audit_file)
    if audit["provenance"].get("dry_run_rows_present"):
        raise SystemExit("audit output contains dry-run rows; refusing to select from it")

    states_file = CT_DIR / f"t1c_group_states.{args.task}.v1.json"
    states_sha = verify_companion_sha256(states_file)
    states = read_json(states_file)
    dh_per_seed = states["D_H"]["per_seed"]
    dh_sum = sum(v["successes"] for v in dh_per_seed.values())
    dh_sumsq = sum(v["successes"] ** 2 for v in dh_per_seed.values())
    if (dh_sum, dh_sumsq) != (T1C_DH_SUM, T1C_DH_SUMSQ):
        raise SystemExit(
            f"frozen T1c D_H sums ({dh_sum}, {dh_sumsq}) != code constants "
            f"({T1C_DH_SUM}, {T1C_DH_SUMSQ})"
        )

    feasible = set(audit["summary"]["feasible_seeds"])
    per_seed_audit = audit["per_seed"]

    def t1c_successes(seed: int) -> int:
        rec = dh_per_seed.get(str(seed))
        return int(rec["successes"]) if rec else 0

    # ---- Expert-Cover-12 ----------------------------------------------------
    cover, slots = select_cover(cells_payload, feasible)
    cover_feasible = len(cover) >= COVER_SIZE
    infeasibility_certificate = None
    if not cover_feasible:
        infeasibility_certificate = {
            "task": args.task,
            "feasible": False,
            "binding_constraints": ["cover_12_distinct_feasible_seeds"],
            "evidence": {"selected": len(cover), "slots": slots,
                         "n_expert_feasible": len(feasible)},
            "consequence": (
                "Cover-12 infeasible; T2 training does not launch under this design "
                "(protocol.t2_expert_audit.v1 infeasibility_outcomes.cover_12)."
            ),
        }

    # ---- dose points ---------------------------------------------------------
    q_points = {}
    if cover_feasible:
        q_points["6"] = cover[:6]
        q_points["12"] = list(cover)
        if len(feasible) >= 20:
            extra = [s for s in cells_payload["global_ranking"]
                     if s in feasible and s not in cover][:8]
            q_points["20"] = list(cover) + extra
        else:
            q_points["20"] = None  # DROPPED, recorded not substituted

    # ---- Expert-Random-12 ----------------------------------------------------
    random_12 = None
    if cover_feasible:
        random_12 = sorted(random.Random(RANDOM_ARM_SEED).sample(sorted(feasible), COVER_SIZE))

    # ---- Self-Diverse-12 -----------------------------------------------------
    dh_manifest = args.t1c_run_dir / "manifest_D_H.jsonl"
    if not dh_manifest.exists():
        raise SystemExit(f"missing T1c D_H manifest: {dh_manifest}")
    self_diverse = select_self_diverse(dh_manifest, dh_per_seed, COVER_SIZE)

    # ---- realized ESS --------------------------------------------------------
    ess = {}
    for q_str, seeds in q_points.items():
        if seeds is None:
            ess[q_str] = {"status": "dropped_insufficient_feasible"}
            continue
        q = len(seeds)
        counts = [t1c_successes(s) for s in seeds]
        n_eff = realized_ess(q, counts)
        n_tail = sum(1 for s in seeds if per_seed_audit[str(s)]["unsupported_tail"])
        ess[q_str] = {
            "q": q,
            "selected_prior_success_counts": counts,
            "n_eff_realized": round(n_eff, 4),
            "n_eff_idealized_all_new": round((T1C_DH_SUM + q) ** 2 / (T1C_DH_SUMSQ + q), 4),
            "ess_rescue": n_eff >= N_EFF_MIN,
            "n_unsupported_tail_selected": n_tail,
        }
    if cover_feasible and ess["12"]["ess_rescue"] and ess["12"]["n_unsupported_tail_selected"] < 7:
        raise SystemExit(
            "arithmetic violation: Q=12 labeled ESS rescue with "
            f"{ess['12']['n_unsupported_tail_selected']} < 7 tail seeds — "
            "contradicts the frozen pre-computed bound; inputs are inconsistent"
        )

    def traj_path(seed: int) -> str | None:
        return per_seed_audit[str(seed)].get("hdf5")

    payload = {
        "record": f"capability_transport.t2_selection.{args.task}.v1",
        "schema_version": 1,
        "protocol_revision": protocol["protocol_revision"],
        "task": args.task,
        "pure_function_note": (
            "Deterministic image of (cells file, audit output, T1c D_H manifest) "
            "under the frozen selection rules; RANDOM_ARM_SEED is the only RNG "
            f"and is frozen at {RANDOM_ARM_SEED}."
        ),
        "arms": {
            "Zero": {"expert_trajectories": []},
            "Self-Diverse-12": {"picks": self_diverse},
            "Expert-Random-12": (
                {"seeds": random_12,
                 "trajectories": {str(s): traj_path(s) for s in random_12}}
                if random_12 is not None else None
            ),
            "Expert-Cover-12": (
                {"seeds": cover, "slots": slots,
                 "trajectories": {str(s): traj_path(s) for s in cover}}
                if cover_feasible else None
            ),
        },
        "dose_points": {
            q: (seeds if seeds is not None else "DROPPED")
            for q, seeds in q_points.items()
        },
        "ess_at_selection": ess,
        "cover_feasible": cover_feasible,
        "infeasibility_certificate": infeasibility_certificate,
        "provenance": {
            "protocol_sha256": protocol_sha,
            "cells_file": rules["cells_file"],
            "cells_sha256": cells_sha,
            "audit_file": rules["output_file"],
            "audit_sha256": audit_sha,
            "t1c_group_states_sha256": states_sha,
            "t1c_dh_manifest": str(dh_manifest),
            "random_arm_seed": RANDOM_ARM_SEED,
        },
    }

    output = args.output or CT_DIR / f"t2_selection.{args.task}.v1.json"
    if output.exists():
        raise SystemExit(f"refusing to overwrite frozen selection {output}")
    write_json_atomic(output, payload)
    out_sha = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(f"{out_sha}  {output.name}\n")
    print(f"wrote {output}\nsha256 {out_sha}")
    print(json.dumps({"cover_feasible": cover_feasible,
                      "cover": cover if cover_feasible else None,
                      "ess": ess}, indent=2))


if __name__ == "__main__":
    main()
