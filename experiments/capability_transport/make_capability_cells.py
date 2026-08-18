#!/usr/bin/env python
"""Freeze capability cells for the 50 hard seeds of place_container_plate.

T2 step 2 of the frozen ordering in protocol.t2_expert_audit.v1.json:
built AFTER that protocol and BEFORE the first expert probe attempt.
Offline, zero training, zero expert attempts: each hard seed's scene is
reconstructed by instantiating the REAL env (setup_demo), never by
reimplementing load_actors — np.random draws before load_actors
(crazy_random_light coin, table_z_bias uniform) shift the RNG stream, so
only the real init path reproduces the frozen scenes.

Frozen cell rule (12 cells):
    cell = (arm_side, container_type, y_band)
      arm_side       = right if container x > 0 else left   (play_once rule)
      container_type = 002_bowl | 021_cup
      y_band         = 3 equal bands over the sampling range y in [-0.10, 0.05):
                       y0 = [-0.10, -0.05), y1 = [-0.05, 0.00), y2 = [0.00, 0.05]
    Geometry is read AFTER setup_demo returns (post-settle) — the state the
    policy actually observes. y_band uses the post-settle container y,
    clamped into the sampling range before banding.

Frozen ranking within a cell (least self-harvestable first):
    (t1c_successes asc, census posterior_mean asc, seed asc)
Unsupported-tail seeds have no T1c attempts: t1c_successes = 0,
posterior_mean = 0.1 (Beta(1, 9) mean) < 0.2 of the 1/8 supported seeds,
so the tail ranks first inside every cell — matching the ESS proof that
any Q = 12 rescue needs >= 7 tail seeds.

Frozen fallback-neighbor order per cell (used when a cell has no
expert-feasible member): adjacent y-band, same arm+type (nearer band
first) -> other container type, same arm+band -> other arm, same
type+band -> everything else by (|band gap|, type mismatch, arm
mismatch, cell key). After all neighbor lists: global frozen ranking.

Usage (inside the cloud container, from /workspace/RoboTwin):
    python experiments/capability_transport/make_capability_cells.py \
        --task place_container_plate --task-config demo_clean
"""

import argparse
import hashlib
import sys
from pathlib import Path

CT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CT_DIR))

from common import (  # noqa: E402
    REPO_ROOT,
    load_task_args,
    make_task_env,
    read_json,
    repo_path,
    write_json_atomic,
)

PROTOCOL_FILE = CT_DIR / "protocol.t2_expert_audit.v1.json"
Y_BANDS = ((-0.10, -0.05), (-0.05, 0.00), (0.00, 0.05))
ARM_SIDES = ("left", "right")
CONTAINER_TYPES = ("002_bowl", "021_cup")


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


def y_band_index(y: float) -> int:
    y = min(max(y, Y_BANDS[0][0]), Y_BANDS[-1][1])
    for idx, (lo, hi) in enumerate(Y_BANDS):
        if y < hi or idx == len(Y_BANDS) - 1:
            return idx
    return len(Y_BANDS) - 1


def cell_key(arm: str, ctype: str, band: int) -> str:
    return f"{arm}|{ctype}|y{band}"


def all_cell_keys():
    return [
        cell_key(arm, ctype, band)
        for arm in ARM_SIDES
        for ctype in CONTAINER_TYPES
        for band in range(len(Y_BANDS))
    ]


def fallback_neighbors(key: str):
    """Frozen deterministic fallback order for one cell (excludes itself)."""
    arm, ctype, band_s = key.split("|")
    band = int(band_s[1:])

    # Tier ordering: same arm+type (adjacent bands first), then same
    # arm+other type, then other arm+same type, then the rest; ties broken
    # by band distance, then cell key.
    def order(other):
        o_arm, o_ctype, o_band_s = other.split("|")
        o_band = int(o_band_s[1:])
        if o_arm == arm and o_ctype == ctype:
            tier = 0
        elif o_arm == arm:
            tier = 1
        elif o_ctype == ctype:
            tier = 2
        else:
            tier = 3
        return (tier, abs(o_band - band), other)

    return sorted((k for k in all_cell_keys() if k != key), key=order)


def extract_geometry(task_name: str, env_args: dict, seed: int) -> dict:
    from envs.utils.create_actor import UnStableError

    env = make_task_env(task_name)
    run_args = dict(env_args)
    run_args["render_freq"] = 0
    unstable_at_init = False
    unstable_detail = None
    try:
        try:
            env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **run_args)
        except UnStableError as exc:
            # load_actors completed before check_stable raised; the actors
            # still exist on the env instance, so geometry is extractable.
            unstable_at_init = True
            unstable_detail = str(exc)
        container_p = [float(v) for v in env.container.get_pose().p]
        plate_p = [float(v) for v in env.plate.get_pose().p]
        arm = "right" if container_p[0] > 0 else "left"
        dx = plate_p[0] - container_p[0]
        dy = plate_p[1] - container_p[1]
        return {
            "seed": seed,
            "container_type": str(env.actor_name),
            "container_id": int(env.container_id),
            "container_xyz": container_p,
            "plate_xyz": plate_p,
            "arm_side": arm,
            "displacement_xy": [dx, dy],
            "planar_distance": float((dx * dx + dy * dy) ** 0.5),
            "table_z_bias": float(env.table_z_bias),
            "crazy_random_light": bool(env.crazy_random_light),
            "unstable_at_init": unstable_at_init,
            "unstable_detail": unstable_detail,
            "y_band": y_band_index(container_p[1]),
        }
    finally:
        try:
            env.close_env()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="place_container_plate")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing frozen output (forbidden after the probe starts)")
    args = parser.parse_args()

    output = args.output or CT_DIR / f"capability_cells.{args.task}.v1.json"
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite frozen output {output} (use --force before the probe only)")

    protocol_sha = verify_companion_sha256(PROTOCOL_FILE)
    protocol = read_json(PROTOCOL_FILE)
    rules = protocol["collector_rules"]
    if rules["task"] != args.task or rules["task_config"] != args.task_config:
        raise SystemExit("task/task-config do not match the frozen audit protocol")

    groups_file = CT_DIR / f"difficulty_groups.{args.task}.v1.json"
    groups_sha = verify_companion_sha256(groups_file)
    groups = read_json(groups_file)
    states_file = CT_DIR / f"t1c_group_states.{args.task}.v1.json"
    states_sha = verify_companion_sha256(states_file)
    states = read_json(states_file)

    tail = {int(s) for s in groups["unsupported_tail_0_of_8"]}
    hard = sorted(int(s) for s, info in groups["seeds"].items() if info["group"] == "hard")
    if len(hard) != rules["candidate_pool_size"]:
        raise SystemExit(f"hard pool size {len(hard)} != frozen {rules['candidate_pool_size']}")
    supported = [s for s in hard if s not in tail]
    dh_per_seed = states["D_H"]["per_seed"]
    if sorted(int(s) for s in dh_per_seed) != supported:
        raise SystemExit("T1c D_H per_seed keys do not match the supported-hard set")

    env_args = load_task_args(args.task, args.task_config)

    records = {}
    for i, seed in enumerate(hard):
        print(f"[{i + 1}/{len(hard)}] reconstructing seed {seed}", flush=True)
        geo = extract_geometry(args.task, env_args, seed)
        info = groups["seeds"][str(seed)]
        t1c = dh_per_seed.get(str(seed))
        geo.update(
            {
                "census_successes": int(info["successes"]),
                "census_repeats": int(info["evaluated_repeats"]),
                "posterior_mean": float(info["posterior_mean"]),
                "unsupported_tail": seed in tail,
                "t1c_attempts": int(t1c["attempts"]) if t1c else 0,
                "t1c_successes": int(t1c["successes"]) if t1c else 0,
                "cell": cell_key(geo["arm_side"], geo["container_type"], geo["y_band"]),
            }
        )
        records[str(seed)] = geo

    cells = {}
    for key in all_cell_keys():
        members = [int(s) for s, r in records.items() if r["cell"] == key]
        ranked = sorted(
            members,
            key=lambda s: (
                records[str(s)]["t1c_successes"],
                records[str(s)]["posterior_mean"],
                s,
            ),
        )
        cells[key] = {
            "members": sorted(members),
            "ranking": ranked,
            "fallback_neighbors": fallback_neighbors(key),
        }

    global_ranking = sorted(
        (int(s) for s in records),
        key=lambda s: (
            records[str(s)]["t1c_successes"],
            records[str(s)]["posterior_mean"],
            s,
        ),
    )

    payload = {
        "record": f"capability_transport.t2_capability_cells.{args.task}.v1",
        "schema_version": 1,
        "protocol_revision": protocol["protocol_revision"],
        "task": args.task,
        "task_config": args.task_config,
        "freeze_note": (
            "Frozen BEFORE the first expert probe attempt per "
            "protocol.t2_expert_audit.v1 freeze_ordering. Cell rule, rankings, "
            "and fallback orders may never change after the probe starts."
        ),
        "cell_rule": {
            "axes": ["arm_side", "container_type", "y_band"],
            "arm_sides": list(ARM_SIDES),
            "container_types": list(CONTAINER_TYPES),
            "y_bands": [list(b) for b in Y_BANDS],
            "geometry_source": "real env setup_demo, post-settle poses (what the policy observes)",
        },
        "ranking_rule": "(t1c_successes asc, posterior_mean asc, seed asc) — least self-harvestable first",
        "n_cells": len(cells),
        "n_seeds": len(records),
        "cells": cells,
        "global_ranking": global_ranking,
        "seeds": records,
        "provenance": {
            "protocol_file": str(PROTOCOL_FILE.relative_to(REPO_ROOT)),
            "protocol_sha256": protocol_sha,
            "groups_file": str(groups_file.relative_to(REPO_ROOT)),
            "groups_file_sha256": groups_sha,
            "t1c_group_states_file": str(states_file.relative_to(REPO_ROOT)),
            "t1c_group_states_sha256": states_sha,
            "task_config_file": f"task_config/{args.task_config}.yml",
            "task_config_sha256": sha256_file(repo_path("task_config", f"{args.task_config}.yml")),
        },
    }

    non_empty = sum(1 for c in cells.values() if c["members"])
    payload["cell_occupancy"] = {
        "non_empty_cells": non_empty,
        "empty_cells": [k for k, c in cells.items() if not c["members"]],
        "sizes": {k: len(c["members"]) for k, c in cells.items()},
    }

    write_json_atomic(output, payload)
    out_sha = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(f"{out_sha}  {output.name}\n")
    print(f"wrote {output}\nsha256 {out_sha}")
    print(f"non-empty cells: {non_empty}/{len(cells)}")
    for k in all_cell_keys():
        print(f"  {k}: {payload['cell_occupancy']['sizes'][k]}")


if __name__ == "__main__":
    main()
