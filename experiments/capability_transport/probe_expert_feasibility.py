#!/usr/bin/env python
"""T2 expert-source feasibility probe for place_container_plate.

Executes the frozen audit of protocol.t2_expert_audit.v1.json (step 3 of its
freeze ordering): for each of the 50 hard seeds, up to ATTEMPT_CAP scripted
expert attempts, stopping the seed at its first FULL success:

    (1) plan_success after play_once() with need_plan=True
    (2) check_success() on the planned execution
    (3) joint trajectory saved (save_traj_data)
    (4) replay materialization: fresh env, need_plan=False, save_data=True,
        set_path_lst from the saved trajectory, play_once(), check_success(),
        merge_pkl_to_hdf5_video() -> episode0.hdf5

Discipline mirrors collect_t1c_sources.py:
  - append-only per-attempt manifest rows (JSONL), per-seed shards supported
    for parallel launch; resume replays recorded rows exactly and refuses to
    resume on any meta / pool / ordering mismatch;
  - the frozen cells file MUST exist (with a valid sha256 companion) before
    the first real attempt — cells are frozen before the probe, per protocol;
  - successful trajectories are materialized under run_dir/trajectories/ and
    PRE-DECLARED to enter the T2 expert pool D_X (no second collection pass);
  - --report merges shards and writes the frozen audit output
    expert_source_feasibility.<task>.v1.json (+ sha256 companion).

There is no policy seed: the scripted expert's only stochasticity is mplib
RRT's internal sampling, which retries sample over. Rows are keyed by
(env_seed, attempt_index).

Usage (inside the cloud container, from /workspace/RoboTwin):
    python experiments/capability_transport/probe_expert_feasibility.py \
        --task place_container_plate --run-dir <dir> [--only-seeds 100002,...]
    python experiments/capability_transport/probe_expert_feasibility.py \
        --task place_container_plate --run-dir <dir> --merge-shards
    python experiments/capability_transport/probe_expert_feasibility.py \
        --task place_container_plate --run-dir <dir> --report
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CT_DIR))

from common import (  # noqa: E402
    REPO_ROOT,
    append_jsonl,
    iter_jsonl,
    load_task_args,
    make_task_env,
    read_json,
    write_json_atomic,
)

# Frozen code constants; validated against protocol collector_rules at startup.
ATTEMPT_CAP = 5
SUCCESS_CAP = 1
UNSTABLE_RETIRE_COUNT = 2
CLEAR_CACHE_FREQ = 5
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_STAGE = "t2_expert_probe"
N_EFF_MIN = 10
T1C_DH_SUM = 41
T1C_DH_SUMSQ = 267

PROTOCOL_FILE = CT_DIR / "protocol.t2_expert_audit.v1.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def load_protocol_rules(task: str, task_config: str):
    protocol_sha = verify_companion_sha256(PROTOCOL_FILE)
    protocol = read_json(PROTOCOL_FILE)
    rules = protocol["collector_rules"]
    checks = [
        ("task", task, rules["task"]),
        ("task_config", task_config, rules["task_config"]),
        ("attempt_cap", ATTEMPT_CAP, int(rules["attempt_cap"])),
        ("success_cap", SUCCESS_CAP, int(rules["success_cap"])),
        ("unstable_retire_count", UNSTABLE_RETIRE_COUNT, int(rules["unstable_retire_count"])),
        ("manifest_schema_version", MANIFEST_SCHEMA_VERSION, int(rules["manifest_schema_version"])),
        ("manifest_stage", MANIFEST_STAGE, rules["manifest_stage"]),
        ("n_eff_min", N_EFF_MIN, int(rules["n_eff_min"])),
        ("t1c_dh_sum", T1C_DH_SUM, int(rules["t1c_dh_sum"])),
        ("t1c_dh_sumsq", T1C_DH_SUMSQ, int(rules["t1c_dh_sumsq"])),
    ]
    for name, code_value, frozen_value in checks:
        if code_value != frozen_value:
            raise SystemExit(
                f"protocol collector_rules.{name} = {frozen_value!r} does not match "
                f"code constant {code_value!r}; refusing to run"
            )
    if not rules["probe_all_candidates"] or not rules["materialization_required"]:
        raise SystemExit("collector_rules flags do not match this collector's semantics")
    return protocol, rules, protocol_sha


def load_frozen_pool(task: str, rules: dict):
    groups_file = CT_DIR / f"difficulty_groups.{task}.v1.json"
    groups_sha = verify_companion_sha256(groups_file)
    groups = read_json(groups_file)
    tail = {int(s) for s in groups["unsupported_tail_0_of_8"]}
    hard = sorted(int(s) for s, info in groups["seeds"].items() if info["group"] == "hard")
    if len(hard) != int(rules["candidate_pool_size"]):
        raise SystemExit(f"hard pool size {len(hard)} != frozen {rules['candidate_pool_size']}")
    return hard, tail, groups_file, groups_sha


def load_frozen_cells(task: str, rules: dict):
    cells_file = REPO_ROOT / rules["cells_file"]
    if not cells_file.exists():
        raise SystemExit(
            f"frozen cells file {cells_file} does not exist; per protocol it must be "
            "frozen BEFORE the first probe attempt (run make_capability_cells.py first)"
        )
    cells_sha = verify_companion_sha256(cells_file)
    return read_json(cells_file), cells_file, cells_sha


class SeedState:
    def __init__(self, env_seed: int):
        self.env_seed = env_seed
        self.attempts = 0
        self.successes = 0
        self.unstable_errors = 0

    @property
    def retired(self) -> bool:
        return self.unstable_errors >= UNSTABLE_RETIRE_COUNT

    @property
    def finished(self) -> bool:
        return (
            self.successes >= SUCCESS_CAP
            or self.attempts >= ATTEMPT_CAP
            or self.retired
        )

    def apply_row(self, row: dict) -> None:
        if int(row["attempt_index"]) != self.attempts:
            raise RuntimeError(
                f"seed {self.env_seed}: manifest attempt_index {row['attempt_index']} "
                f"!= expected {self.attempts} (append-only order violated)"
            )
        if self.finished:
            raise RuntimeError(
                f"seed {self.env_seed}: manifest row beyond the frozen per-seed budget"
            )
        self.attempts += 1
        if row.get("passed"):
            self.successes += 1
        if row.get("error_type") == "UnStableError":
            self.unstable_errors += 1

    def verdict(self) -> dict:
        if self.successes >= 1:
            status = "expert_feasible"
        elif self.retired:
            status = "env_unstable"
        elif self.attempts >= ATTEMPT_CAP:
            status = "not_harvestable_under_frozen_budget"
        else:
            status = "collecting"
        out = {
            "attempts": self.attempts,
            "successes": self.successes,
            "unstable_errors": self.unstable_errors,
            "status": status,
        }
        if status == "not_harvestable_under_frozen_budget":
            # one-sided bound, never a claim that p = 0
            out["p_upper_95"] = round(1.0 - 0.05 ** (1.0 / self.attempts), 6)
        return out


def manifest_path(run_dir: Path, seed: int | None = None) -> Path:
    if seed is not None:
        return run_dir / f"manifest_X.seed_{int(seed)}.jsonl"
    return run_dir / "manifest_X.jsonl"


def load_recorded_rows(path: Path, pool, meta: dict) -> dict:
    pool_set = set(pool)
    recorded = {}
    if not path.exists():
        return recorded
    for index, row in enumerate(iter_jsonl(path)):
        for key, expected in meta.items():
            if row.get(key) != expected:
                raise RuntimeError(
                    f"Refusing to resume {path} row {index}: {key} is "
                    f"{row.get(key)!r}, expected {expected!r}"
                )
        env_seed = int(row["env_seed"])
        if env_seed not in pool_set:
            raise RuntimeError(
                f"Manifest row {index} in {path}: env_seed {env_seed} outside the frozen pool"
            )
        key = (env_seed, int(row["attempt_index"]))
        if key in recorded:
            raise RuntimeError(f"Manifest row {index}: duplicate attempt {key}")
        recorded[key] = row
    return recorded


def expert_attempt_once(task_name: str, env_args: dict, env_seed: int, tmp_dir: Path,
                        clear_cache: bool) -> dict:
    """One full expert attempt: plan -> execute-check -> save -> replay-materialize."""
    from envs.utils.create_actor import UnStableError

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "plan_success": False,
        "exec_check_success": False,
        "trajectory_saved": False,
        "replay_check_success": False,
        "materialized": False,
        "passed": False,
        "error_type": None,
        "error_message": None,
        "raw_hdf5": None,
        "raw_video": None,
    }

    # ---- Phase A: plan + execute + save trajectory -------------------------
    plan_args = dict(env_args)
    plan_args.update({
        "need_plan": True,
        "save_data": False,
        "save_path": str(tmp_dir),
        "render_freq": 0,
    })
    env = make_task_env(task_name)
    try:
        env.setup_demo(now_ep_num=0, seed=env_seed, **plan_args)
        env.play_once()
        result["plan_success"] = bool(env.plan_success)
        if not result["plan_success"]:
            result["error_type"] = "plan_failed"
            return result
        result["exec_check_success"] = bool(env.check_success())
        if not result["exec_check_success"]:
            result["error_type"] = "execution_check_failed"
            return result
        env.save_traj_data(0)
        result["trajectory_saved"] = True
    except UnStableError as exc:
        result["error_type"] = "UnStableError"
        result["error_message"] = str(exc)
        return result
    except Exception as exc:  # noqa: BLE001 — taxonomy row, then continue the audit
        result["error_type"] = "env_error"
        result["error_message"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        try:
            env.close_env()
        except Exception:
            pass

    # ---- Phase B: replay materialization ------------------------------------
    replay_args = dict(env_args)
    replay_args.update({
        "need_plan": False,
        "save_data": True,
        "save_path": str(tmp_dir),
        "render_freq": 0,
        "left_joint_path": [],
        "right_joint_path": [],
    })
    if replay_args.get("save_freq") is None:
        replay_args["save_freq"] = 15
    env = make_task_env(task_name)
    try:
        env.setup_demo(now_ep_num=0, seed=env_seed, **replay_args)
        traj_data = env.load_tran_data(0)
        replay_args["left_joint_path"] = traj_data["left_joint_path"]
        replay_args["right_joint_path"] = traj_data["right_joint_path"]
        env.set_path_lst(replay_args)
        env.play_once()
        result["replay_check_success"] = bool(env.check_success())
        if not result["replay_check_success"]:
            result["error_type"] = "materialization_failed"
            result["error_message"] = "replay executed but check_success() was False"
            return result
        env.merge_pkl_to_hdf5_video()
        raw_hdf5 = tmp_dir / "data" / "episode0.hdf5"
        if not raw_hdf5.is_file():
            result["error_type"] = "materialization_failed"
            result["error_message"] = f"replay succeeded but no hdf5 at {raw_hdf5}"
            return result
        result["materialized"] = True
        result["raw_hdf5"] = raw_hdf5
        raw_video = tmp_dir / "video" / "episode0.mp4"
        result["raw_video"] = raw_video if raw_video.is_file() else None
        result["passed"] = True
        return result
    except UnStableError as exc:
        result["error_type"] = "UnStableError"
        result["error_message"] = str(exc)
        return result
    except Exception as exc:  # noqa: BLE001
        result["error_type"] = "materialization_failed"
        result["error_message"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        try:
            env.remove_data_cache()
        except Exception:
            pass
        try:
            env.close_env(clear_cache=clear_cache)
        except Exception:
            pass


def dry_run_attempt(env_seed: int, attempt_index: int) -> dict:
    """Deterministic hash draw, logic/resume smoke test ONLY (never predictive)."""
    digest = hashlib.sha256(f"t2probe:{env_seed}:{attempt_index}".encode()).hexdigest()
    draw = int(digest[:8], 16) / 0xFFFFFFFF
    passed = draw < 0.7
    return {
        "plan_success": passed or draw < 0.85,
        "exec_check_success": passed,
        "trajectory_saved": passed,
        "replay_check_success": passed,
        "materialized": passed,
        "passed": passed,
        "error_type": None if passed else ("plan_failed" if draw >= 0.85 else "execution_check_failed"),
        "error_message": None,
        "raw_hdf5": None,
        "raw_video": None,
    }


def materialize_success(run_dir: Path, env_seed: int, result: dict):
    hdf5_rel = None
    video_rel = None
    if result["passed"] and result["raw_hdf5"] is not None:
        dest_dir = run_dir / "trajectories" / f"seed_{env_seed}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "episode0.hdf5"
        shutil.move(str(result["raw_hdf5"]), str(dest))
        hdf5_rel = str(dest.relative_to(run_dir))
        if result["raw_video"] is not None:
            vdest = dest_dir / "episode0.mp4"
            shutil.move(str(result["raw_video"]), str(vdest))
            video_rel = str(vdest.relative_to(run_dir))
    return hdf5_rel, video_rel


def build_meta(args, protocol_sha: str, cells_sha: str, groups_sha: str) -> dict:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage": MANIFEST_STAGE,
        "task": args.task,
        "task_config": args.task_config,
        "protocol_sha256": protocol_sha,
        "cells_sha256": cells_sha,
        "groups_sha256": groups_sha,
    }


def probe(args, protocol_sha, cells_sha, groups_sha, pool, run_dir: Path) -> None:
    seeds = pool
    if args.only_seeds:
        wanted = sorted({int(s) for s in args.only_seeds.split(",")})
        outside = [s for s in wanted if s not in set(pool)]
        if outside:
            raise SystemExit(f"--only-seeds outside the frozen pool: {outside}")
        seeds = wanted
    shard = len(seeds) < len(pool)
    meta = build_meta(args, protocol_sha, cells_sha, groups_sha)

    env_args = None
    if not args.dry_run:
        env_args = load_task_args(args.task, args.task_config)

    def build_states(recorded):
        states = {}
        for (s, a) in sorted(recorded):
            states.setdefault(s, SeedState(s)).apply_row(recorded[(s, a)])
        return states

    shared_states = None
    if not shard:
        mpath_shared = manifest_path(run_dir)
        recorded = load_recorded_rows(mpath_shared, pool, meta)
        if recorded and not args.resume:
            raise SystemExit(
                f"{mpath_shared} already has rows; pass --resume to continue (append-only)"
            )
        shared_states = build_states(recorded)

    live_attempts = 0
    for env_seed in seeds:
        if shard:
            mpath = manifest_path(run_dir, env_seed)
            recorded = load_recorded_rows(mpath, pool, meta)
            if recorded and not args.resume:
                raise SystemExit(
                    f"{mpath} already has rows; pass --resume to continue (append-only)"
                )
            state = build_states(recorded).get(env_seed, SeedState(env_seed))
        else:
            mpath = manifest_path(run_dir)
            state = shared_states.get(env_seed, SeedState(env_seed))
        while not state.finished:
            attempt_index = state.attempts
            started = utc_now()
            t0 = time.time()
            if args.dry_run:
                result = dry_run_attempt(env_seed, attempt_index)
            else:
                tmp_dir = run_dir / ".tmp" / f"X_{env_seed}_a{attempt_index}"
                live_attempts += 1
                result = expert_attempt_once(
                    args.task, env_args, env_seed, tmp_dir,
                    clear_cache=(live_attempts % CLEAR_CACHE_FREQ == 0),
                )
            hdf5_rel, video_rel = (None, None)
            if not args.dry_run:
                hdf5_rel, video_rel = materialize_success(run_dir, env_seed, result)
                shutil.rmtree(run_dir / ".tmp" / f"X_{env_seed}_a{attempt_index}",
                              ignore_errors=True)
            row = {
                **meta,
                "created_at": started,
                "env_seed": env_seed,
                "attempt_index": attempt_index,
                "plan_success": bool(result["plan_success"]),
                "exec_check_success": bool(result["exec_check_success"]),
                "trajectory_saved": bool(result["trajectory_saved"]),
                "replay_check_success": bool(result["replay_check_success"]),
                "materialized": bool(result["materialized"]),
                "passed": bool(result["passed"]),
                "error_type": result["error_type"],
                "error_message": result["error_message"],
                "hdf5": hdf5_rel,
                "video": video_rel,
                "wall_seconds": round(time.time() - t0, 2),
                "dry_run": bool(args.dry_run),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "pid": os.getpid(),
            }
            append_jsonl(mpath, row)
            state.apply_row(row)
            print(
                f"seed {env_seed} attempt {attempt_index}: "
                f"{'PASS' if row['passed'] else row['error_type']}",
                flush=True,
            )
        print(f"seed {env_seed} finished: {json.dumps(state.verdict())}", flush=True)


def merge_shards(pool, run_dir: Path, meta: dict) -> None:
    merged = manifest_path(run_dir)
    if merged.exists():
        raise SystemExit(f"{merged} already exists; refusing to overwrite the merged manifest")
    rows = []
    for env_seed in pool:
        shard = manifest_path(run_dir, env_seed)
        if shard.exists():
            rows.extend(iter_jsonl(shard))
    rows.sort(key=lambda r: (int(r["env_seed"]), int(r["attempt_index"])))
    for row in rows:
        append_jsonl(merged, row)
    # validate the merged stream end-to-end
    recorded = load_recorded_rows(merged, pool, meta)
    states = {}
    for (s, a) in sorted(recorded):
        states.setdefault(s, SeedState(s)).apply_row(recorded[(s, a)])
    print(f"merged {len(rows)} rows for {len(states)} seeds into {merged}")


def write_report(args, protocol, rules, protocol_sha, cells_payload, cells_sha,
                 groups_sha, pool, tail, run_dir: Path) -> None:
    meta = build_meta(args, protocol_sha, cells_sha, groups_sha)
    merged = manifest_path(run_dir)
    if not merged.exists():
        raise SystemExit(f"no merged manifest at {merged}; run --merge-shards first")
    recorded = load_recorded_rows(merged, pool, meta)
    states = {seed: SeedState(seed) for seed in pool}
    success_rows = {}
    for (s, a) in sorted(recorded):
        row = recorded[(s, a)]
        states[s].apply_row(row)
        if row.get("passed"):
            success_rows[s] = row
    unfinished = [s for s, st in states.items() if not st.finished]
    if unfinished:
        raise SystemExit(f"probe incomplete for seeds {unfinished}; refusing to write the audit")

    per_seed = {}
    for seed in pool:
        v = states[seed].verdict()
        v["unsupported_tail"] = seed in tail
        v["cell"] = cells_payload["seeds"][str(seed)]["cell"]
        if seed in success_rows:
            v["hdf5"] = success_rows[seed]["hdf5"]
            v["video"] = success_rows[seed]["video"]
        per_seed[str(seed)] = v

    feasible = sorted(s for s in pool if per_seed[str(s)]["status"] == "expert_feasible")
    cells_out = {}
    for key, cell in cells_payload["cells"].items():
        feas_members = [s for s in cell["ranking"] if s in set(feasible)]
        cells_out[key] = {
            "members": cell["members"],
            "feasible_ranked": feas_members,
            "expert_feasible": bool(feas_members),
        }
    n_cells_feasible = sum(1 for c in cells_out.values() if c["expert_feasible"])

    ess = {
        str(q): round((T1C_DH_SUM + q) ** 2 / (T1C_DH_SUMSQ + q), 4)
        for q in rules["q_dose_points"]
    }
    q_feasibility = {
        str(q): len(feasible) >= q for q in rules["q_dose_points"]
    }
    cover_feasible = len(feasible) >= int(rules["min_feasible_for_cover"])

    report = {
        "record": f"capability_transport.t2_expert_audit.{args.task}.v1",
        "schema_version": 1,
        "protocol_revision": protocol["protocol_revision"],
        "task": args.task,
        "task_config": args.task_config,
        "pool": rules["candidate_pool"],
        "attempt_cap": ATTEMPT_CAP,
        "success_cap": SUCCESS_CAP,
        "per_seed": per_seed,
        "cells": cells_out,
        "summary": {
            "n_probed": len(pool),
            "n_expert_feasible": len(feasible),
            "n_env_unstable": sum(1 for s in pool if per_seed[str(s)]["status"] == "env_unstable"),
            "n_not_harvestable": sum(
                1 for s in pool
                if per_seed[str(s)]["status"] == "not_harvestable_under_frozen_budget"
            ),
            "n_cells_with_feasible_member": n_cells_feasible,
            "n_cells_total": cells_payload["n_cells"],
            "feasible_seeds": feasible,
            "feasible_tail_seeds": sorted(s for s in feasible if s in tail),
            "feasible_supported_seeds": sorted(s for s in feasible if s not in tail),
        },
        "design_feasibility": {
            "cover_12": cover_feasible,
            "q_dose_points": q_feasibility,
            "ess_at_q": ess,
            "ess_rule": rules["ess_bound"],
        },
        "language_note": (
            "0-success seeds are 'not harvestable under the frozen budget' with the "
            "recorded one-sided p_upper_95; this audit never claims p = 0."
        ),
        "provenance": {
            "protocol_file": str(PROTOCOL_FILE.relative_to(REPO_ROOT)),
            "protocol_sha256": protocol_sha,
            "cells_file": rules["cells_file"],
            "cells_sha256": cells_sha,
            "groups_sha256": groups_sha,
            "run_dir": str(run_dir),
            "manifest": str(merged.relative_to(run_dir)),
            "dry_run_rows_present": any(r.get("dry_run") for r in recorded.values()),
        },
    }
    if not cover_feasible:
        report["infeasibility_certificate"] = {
            "task": args.task,
            "feasible": False,
            "binding_constraints": ["min_feasible_for_cover_12"],
            "evidence": {
                "n_expert_feasible": len(feasible),
                "n_cells_with_feasible_member": n_cells_feasible,
            },
            "consequence": (
                "Cover-12 infeasible under the frozen probe budget; T2 training does "
                "not launch under this design (protocol.t2_expert_audit.v1 "
                "infeasibility_outcomes.cover_12). No cap increases or re-probes."
            ),
        }

    is_dry = any(r.get("dry_run") for r in recorded.values())
    if is_dry:
        out = run_dir / f"expert_source_feasibility.{args.task}.DRYRUN.json"
    else:
        out = REPO_ROOT / rules["output_file"]
        if out.exists():
            raise SystemExit(f"refusing to overwrite frozen audit output {out}")
    write_json_atomic(out, report)
    out_sha = sha256_file(out)
    out.with_suffix(out.suffix + ".sha256").write_text(f"{out_sha}  {out.name}\n")
    print(f"wrote {out}\nsha256 {out_sha}")
    print(json.dumps(report["summary"], indent=2))

    if not is_dry:
        attempts_total = len(recorded)
        append_jsonl(CT_DIR / "efficiency_ledger.jsonl", {
            "created_at": utc_now(),
            "record": report["record"],
            "task": args.task,
            "stage": MANIFEST_STAGE,
            "attempts": attempts_total,
            "successes": len(feasible),
            "run_dir": str(run_dir),
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="place_container_plate")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--only-seeds", default=None,
                        help="comma-separated env seeds; enables per-seed manifest shards")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="deterministic hash draws; logic/resume smoke test only")
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    protocol, rules, protocol_sha = load_protocol_rules(args.task, args.task_config)
    pool, tail, _groups_file, groups_sha = load_frozen_pool(args.task, rules)
    cells_payload, _cells_file, cells_sha = load_frozen_cells(args.task, rules)

    cell_seeds = sorted(int(s) for s in cells_payload["seeds"])
    if cell_seeds != pool:
        raise SystemExit("cells file seed set does not match the frozen hard pool")

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_shards:
        merge_shards(pool, run_dir, build_meta(args, protocol_sha, cells_sha, groups_sha))
        return
    if args.report:
        write_report(args, protocol, rules, protocol_sha, cells_payload, cells_sha,
                     groups_sha, pool, tail, run_dir)
        return
    probe(args, protocol_sha, cells_sha, groups_sha, pool, run_dir)


if __name__ == "__main__":
    main()
