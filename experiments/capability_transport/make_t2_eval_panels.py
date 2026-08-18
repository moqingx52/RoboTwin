#!/usr/bin/env python3
"""Freeze T2 independent eval panels from a completed 200×8 π₀ census.

Reads the merged eval result, assigns Beta-Binomial E/M/H groups and
capability cells (same geometry rule as make_capability_cells.py), checks
the identifiability floors in protocol.t2_eval_census.v1.json, and writes
either t2_eval_panels.place_container_plate.v1.json (+ sha256) or the
infeasibility certificate t2_eval_unidentifiable.place_container_plate.v1.json.

Usage (inside the cloud container, from /workspace/RoboTwin):
    python experiments/capability_transport/make_t2_eval_panels.py \
        --result experiments/capability_transport/runs/<run>/place_container_plate/t2_eval_census_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

CT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CT_DIR))

from common import load_task_args, make_task_env, read_json, repo_path, write_json_atomic  # noqa: E402
from make_capability_cells import (  # noqa: E402
    cell_key,
    extract_geometry,
    verify_companion_sha256,
)
from make_difficulty_groups import (  # noqa: E402
    EXPECTED_REPEATS,
    MIN_EVALUATED_REPEATS,
    assign_group,
    posterior_mean,
)

PROTOCOL_FILE = CT_DIR / "protocol.t2_eval_census.v1.json"
SELECTION_FILE = CT_DIR / "t2_selection.place_container_plate.v1.json"
CELLS_FILE = CT_DIR / "capability_cells.place_container_plate.v1.json"
GROUPS_FILE = CT_DIR / "difficulty_groups.place_container_plate.v1.json"
SPLIT_NAME = "t2_eval_census"

COVER12_OCCUPIED_SOURCE_CELLS = (
    "left|021_cup|y2",
    "right|002_bowl|y0",
    "right|002_bowl|y1",
    "right|002_bowl|y2",
    "right|021_cup|y1",
)
DOMINANT_TRANSPORT_CELLS = ("right|002_bowl|y1", "right|002_bowl|y2")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_per_seed(rows: list[dict], split: str) -> dict[int, dict]:
    per_seed: dict[int, dict] = {}
    for row in rows:
        if row["split"] != split:
            continue
        seed = int(row["env_seed"])
        entry = per_seed.setdefault(seed, {"successes": 0, "evaluated": 0, "missing": 0})
        if row.get("evaluated", True):
            entry["evaluated"] += 1
            entry["successes"] += int(bool(row["success"]))
        else:
            entry["missing"] += 1
    return per_seed


def build_groups(per_seed: dict[int, dict]) -> tuple[dict, list[int], list[int]]:
    groups = {"easy": [], "medium": [], "hard": []}
    unsupported_tail: list[int] = []
    excluded: list[int] = []
    seed_records: dict[str, dict] = {}

    for seed in sorted(per_seed):
        entry = per_seed[seed]
        total = entry["evaluated"] + entry["missing"]
        if total != EXPECTED_REPEATS:
            raise SystemExit(f"Seed {seed} has {total} rows, expected {EXPECTED_REPEATS}")
        record = {
            "successes": entry["successes"],
            "evaluated_repeats": entry["evaluated"],
            "operational_missing_repeats": entry["missing"],
        }
        if entry["evaluated"] < MIN_EVALUATED_REPEATS:
            record["group"] = None
            record["excluded_reason"] = f"evaluated_repeats < {MIN_EVALUATED_REPEATS}"
            excluded.append(seed)
        else:
            mean = posterior_mean(entry["successes"], entry["evaluated"])
            group = assign_group(mean)
            record["posterior_mean"] = round(mean, 6)
            record["group"] = group
            groups[group].append(seed)
            if entry["successes"] == 0 and entry["evaluated"] == EXPECTED_REPEATS:
                unsupported_tail.append(seed)
        seed_records[str(seed)] = record

    return (
        {
            "groups": {name: sorted(seeds) for name, seeds in groups.items()},
            "group_sizes": {name: len(seeds) for name, seeds in groups.items()},
            "unsupported_tail_0_of_8": sorted(unsupported_tail),
            "excluded_insufficient_repeats": sorted(excluded),
            "seeds": seed_records,
        },
        unsupported_tail,
        excluded,
    )


def _record_from_geo(geo: dict) -> dict:
    return {
        "cell": cell_key(geo["arm_side"], geo["container_type"], geo["y_band"]),
        "arm_side": geo["arm_side"],
        "container_type": geo["container_type"],
        "y_band": geo["y_band"],
        "unstable_at_init": geo["unstable_at_init"],
    }


def _extract_on_env(env, env_args: dict, seed: int) -> dict:
    """Same geometry rule as extract_geometry, reusing one env per worker."""
    from envs.utils.create_actor import UnStableError
    from make_capability_cells import y_band_index

    run_args = dict(env_args)
    run_args["render_freq"] = 0
    unstable_at_init = False
    try:
        env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **run_args)
    except UnStableError:
        unstable_at_init = True
    container_p = [float(v) for v in env.container.get_pose().p]
    plate_p = [float(v) for v in env.plate.get_pose().p]
    arm = "right" if container_p[0] > 0 else "left"
    return {
        "arm_side": arm,
        "container_type": str(env.actor_name),
        "y_band": y_band_index(container_p[1]),
        "unstable_at_init": unstable_at_init,
    }


def _extract_shard(payload: tuple) -> dict[str, dict]:
    gpu_id, shard_id, n_shards, seeds, task, task_config = payload
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env_args = load_task_args(task, task_config)
    records: dict[str, dict] = {}
    env = make_task_env(task)
    try:
        for i, seed in enumerate(seeds):
            print(
                f"[gpu {gpu_id} shard {shard_id}/{n_shards}] "
                f"{i + 1}/{len(seeds)} seed {seed}",
                flush=True,
            )
            try:
                geo = _extract_on_env(env, env_args, seed)
            except Exception:
                geo = extract_geometry(task, env_args, seed)
            records[str(seed)] = _record_from_geo(geo)
    finally:
        try:
            env.close_env()
        except Exception:
            pass
    return records


def build_cell_map(
    task: str,
    task_config: str,
    seeds: list[int],
    gpu_ids: list[int],
    workers_per_gpu: int,
) -> dict[str, dict]:
    n_workers = max(1, len(gpu_ids) * workers_per_gpu)
    if n_workers == 1:
        env_args = load_task_args(task, task_config)
        records: dict[str, dict] = {}
        for i, seed in enumerate(seeds):
            print(f"[{i + 1}/{len(seeds)}] cell geometry seed {seed}", flush=True)
            records[str(seed)] = _record_from_geo(extract_geometry(task, env_args, seed))
        return records

    shards = [seeds[i::n_workers] for i in range(n_workers)]
    jobs = []
    for shard_id, shard_seeds in enumerate(shards):
        if not shard_seeds:
            continue
        gpu_id = gpu_ids[shard_id // workers_per_gpu]
        jobs.append((gpu_id, shard_id, n_workers, shard_seeds, task, task_config))

    ctx = mp.get_context("spawn")
    print(
        f"Extracting geometry for {len(seeds)} seeds with {len(jobs)} workers "
        f"({len(gpu_ids)} GPUs × {workers_per_gpu})",
        flush=True,
    )
    with ctx.Pool(len(jobs)) as pool:
        parts = pool.map(_extract_shard, jobs)
    records: dict[str, dict] = {}
    for part in parts:
        records.update(part)
    missing = [s for s in seeds if str(s) not in records]
    if missing:
        raise SystemExit(f"geometry missing for {len(missing)} seeds: {missing[:10]}")
    return records


def panel_counts(hard_seeds: list[int], cell_map: dict[str, dict]) -> dict[str, int]:
    within = [
        s for s in hard_seeds
        if cell_map[str(s)]["cell"] in COVER12_OCCUPIED_SOURCE_CELLS
    ]
    cross = [s for s in hard_seeds if s not in within]
    dominant = [
        s for s in hard_seeds
        if cell_map[str(s)]["cell"] in DOMINANT_TRANSPORT_CELLS
    ]
    return {
        "hard_total": len(hard_seeds),
        "within_cell_hard": len(within),
        "cross_cell_hard": len(cross),
        "right_bowl_y1_y2_hard": len(dominant),
    }


def check_floors(protocol: dict, groups: dict, counts: dict) -> list[str]:
    floors = protocol["identifiability_floors"]["floors"]
    min_repeats = floors["min_evaluated_repeats_per_seed"]
    failures: list[str] = []

    def eligible(name: str) -> int:
        return sum(
            1
            for seed in groups["groups"][name]
            if groups["seeds"][str(seed)]["evaluated_repeats"] >= min_repeats
        )

    for panel in ("easy", "medium", "hard"):
        key = f"{panel}_panel_min_seeds"
        got = eligible(panel)
        need = floors[key]
        if got < need:
            failures.append(f"{key}: {got} < {need}")

    for key, got_key in (
        ("cover12_within_cell_hard_min", "within_cell_hard"),
        ("cover12_cross_cell_hard_min", "cross_cell_hard"),
        ("right_bowl_y1_y2_hard_min", "right_bowl_y1_y2_hard"),
    ):
        got = counts[got_key]
        need = floors[key]
        if got < need:
            failures.append(f"{key}: {got} < {need}")

    return failures


def memorization_seeds() -> list[int]:
    groups = read_json(GROUPS_FILE)
    return sorted(int(s) for s in groups["groups"]["hard"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True, help="Merged t2_eval_census_v1.json")
    parser.add_argument("--output", type=Path, default=None, help="Panels output (default from protocol)")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing frozen output")
    parser.add_argument(
        "--gpu-ids",
        nargs="*",
        type=int,
        default=None,
        help="Physical GPU IDs for geometry workers (default: 0-7)",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=3,
        help="Simulator geometry workers per GPU (measured packing: 3)",
    )
    args = parser.parse_args()

    protocol_sha = verify_companion_sha256(PROTOCOL_FILE)
    protocol = read_json(PROTOCOL_FILE)
    selection_sha = verify_companion_sha256(SELECTION_FILE)
    cells_sha = verify_companion_sha256(CELLS_FILE)

    payload = read_json(args.result)
    if not payload.get("progress", {}).get("complete"):
        raise SystemExit(f"Result is not complete: {args.result}")

    task = payload["task_name"]
    if task != protocol["collector_rules"]["task"]:
        raise SystemExit(f"task mismatch: {task} != {protocol['collector_rules']['task']}")

    per_seed = aggregate_per_seed(payload["rows"], SPLIT_NAME)
    expected = protocol["collector_rules"]["seed_count"]
    if len(per_seed) != expected:
        raise SystemExit(f"census seed count {len(per_seed)} != expected {expected}")

    group_payload, unsupported_tail, excluded = build_groups(per_seed)
    gpu_ids = args.gpu_ids if args.gpu_ids else [0, 1, 2, 3, 4, 5, 6, 7]
    cell_map = build_cell_map(
        task,
        protocol["collector_rules"]["task_config"],
        sorted(per_seed),
        gpu_ids,
        args.workers_per_gpu,
    )

    hard_panel = group_payload["groups"]["hard"]
    counts = panel_counts(hard_panel, cell_map)
    within_hard = [
        s for s in hard_panel if cell_map[str(s)]["cell"] in COVER12_OCCUPIED_SOURCE_CELLS
    ]
    cross_hard = [s for s in hard_panel if s not in within_hard]
    dominant_hard = [
        s for s in hard_panel if cell_map[str(s)]["cell"] in DOMINANT_TRANSPORT_CELLS
    ]

    floor_failures = check_floors(protocol, group_payload, counts)
    result_sha = sha256_file(args.result)

    if floor_failures:
        cert_path = CT_DIR / "t2_eval_unidentifiable.place_container_plate.v1.json"
        if cert_path.exists() and not args.force:
            raise SystemExit(f"Refusing to overwrite existing certificate: {cert_path}")
        cert = {
            "record": "capability_transport.t2_eval_unidentifiable.place_container_plate.v1",
            "schema_version": 1,
            "protocol_revision": protocol["protocol_revision"],
            "identifiable": False,
            "floor_failures": floor_failures,
            "panel_counts": counts,
            "group_sizes": group_payload["group_sizes"],
            "source_result": str(args.result),
            "source_result_sha256": result_sha,
            "action": protocol["infeasibility_outcome"]["action"],
            "provenance": {
                "protocol_file": str(PROTOCOL_FILE.relative_to(repo_path())),
                "protocol_sha256": protocol_sha,
            },
        }
        write_json_atomic(cert_path, cert)
        cert_sha = sha256_file(cert_path)
        cert_path.with_suffix(cert_path.suffix + ".sha256").write_text(
            f"{cert_sha}  {cert_path.name}\n"
        )
        print(f"IDENTIFIABILITY FAILED — wrote {cert_path}")
        for item in floor_failures:
            print(f"  - {item}")
        raise SystemExit(1)

    output = args.output or repo_path(protocol["collector_rules"]["output_panels"])
    if output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite frozen panels: {output} (use --force to replace)")

    out = {
        "record": "capability_transport.t2_eval_panels.place_container_plate.v1",
        "schema_version": 1,
        "protocol_revision": protocol["protocol_revision"],
        "task": task,
        "identifiable": True,
        "source_result": str(args.result),
        "source_result_sha256": result_sha,
        "expected_repeats": EXPECTED_REPEATS,
        "min_evaluated_repeats": MIN_EVALUATED_REPEATS,
        "posterior": "p0 | s ~ Beta(1 + s, 1 + R_eval - s); grouped by posterior mean",
        "group_bounds": protocol["difficulty_grouping"]["groups"],
        "panels": {
            "easy": group_payload["groups"]["easy"],
            "medium": group_payload["groups"]["medium"],
            "hard": group_payload["groups"]["hard"],
            "within_cell_hard": sorted(within_hard),
            "cross_cell_hard": sorted(cross_hard),
            "right_bowl_y1_y2_hard": sorted(dominant_hard),
            "memorization_diagnostic_hard": memorization_seeds(),
        },
        "panel_sizes": {
            "easy": len(group_payload["groups"]["easy"]),
            "medium": len(group_payload["groups"]["medium"]),
            "hard": len(hard_panel),
            "within_cell_hard": len(within_hard),
            "cross_cell_hard": len(cross_hard),
            "right_bowl_y1_y2_hard": len(dominant_hard),
            "memorization_diagnostic_hard": len(memorization_seeds()),
        },
        "identifiability_floors_met": protocol["identifiability_floors"]["floors"],
        "panel_counts_at_freeze": counts,
        "unsupported_tail_0_of_8": group_payload["unsupported_tail_0_of_8"],
        "excluded_insufficient_repeats": group_payload["excluded_insufficient_repeats"],
        "seeds": {
            seed: {
                **group_payload["seeds"][seed],
                **cell_map[seed],
            }
            for seed in group_payload["seeds"]
        },
        "cover12_occupied_source_cells": list(COVER12_OCCUPIED_SOURCE_CELLS),
        "freeze_note": (
            "Frozen from the independent 200×8 π₀ census only. Primary Δ_H and Γ use "
            "panels.hard (and E/M regression panels) exclusively. Memorization diagnostic "
            "uses the original 50 T1b hard seeds and must not enter primary inference."
        ),
        "provenance": {
            "protocol_file": str(PROTOCOL_FILE.relative_to(repo_path())),
            "protocol_sha256": protocol_sha,
            "selection_file": str(SELECTION_FILE.relative_to(repo_path())),
            "selection_sha256": selection_sha,
            "cells_rule_file": str(CELLS_FILE.relative_to(repo_path())),
            "cells_rule_sha256": cells_sha,
            "ckpt_path": payload["ckpt_path"],
            "policy_seed_offset": payload["progress"]["policy_seed_offset"],
        },
    }

    write_json_atomic(output, out)
    out_sha = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(f"{out_sha}  {output.name}\n")
    print(f"Wrote {output}\nsha256 {out_sha}")
    print("Panel sizes:", out["panel_sizes"])


if __name__ == "__main__":
    mp.freeze_support()
    main()
