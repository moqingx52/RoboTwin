#!/usr/bin/env python3
"""T1c source acquisition: round-robin success-first collection with pi0.

Implements protocol.t1.v1.1.json t1c_data_sources_v1_1 for tasks that passed
the source-feasibility gate (t1c_source_feasibility.v1.json).

Frozen semantics (do not change without a protocol amendment):
  - Candidate pools derive deterministically from the frozen
    difficulty_groups.<task>.v1.json (sha256-checked against its companion):
    D_E = the 30 lowest-numbered easy seeds, D_M = all medium seeds,
    D_H = supported-hard (hard minus the 0/8 unsupported tail).
  - Traversal: rounds r = 0,1,2,...; each round attempts every candidate seed
    once in ascending seed order, skipping finished/retired seeds; the group
    stop condition is evaluated before every attempt.
  - policy_seed = POLICY_SEED_BASE + attempt_index_within_seed. Every attempt
    (including UnStableError) consumes one attempt_index.
  - A seed is finished when attempts >= per_seed_attempt_cap or stored
    trajectories >= ceil(0.10 * N_target); retired after 2 UnStableErrors.
  - Store ONLY complete successful trajectories (hdf5 via
    merge_pkl_to_hdf5_video); every attempt (success, failure, UnStableError)
    is appended to the group manifest and never deleted.
  - Stop: N_target successes AND U_h >= U_min -> complete; every candidate
    seed finished/retired first -> source_infeasible (halts T1d for the task).

Resume is exact: recorded manifest rows are replayed THROUGH the frozen
traversal simulation, and live collection begins at the first attempt the
traversal requests that has no recorded outcome. The resumed realization is
therefore identical to an uninterrupted serial run.
"""
import argparse
import hashlib
import math
import os
import shutil
import sys
import time
from pathlib import Path

from common import (
    append_jsonl,
    iter_jsonl,
    load_task_args,
    make_task_env,
    read_json,
    repo_path,
    write_json_atomic,
)

sys.path.append(str(repo_path()))

# Frozen in protocol.t1.v1.1.json t1c_data_sources_v1_1.acquisition_rule:
# offset 6000 is disjoint from T1b 5000..5007, legacy census 3000+, phase1 0+.
POLICY_SEED_BASE = 6000
# "m_s <= ceil(0.10 * N_target_group)" (single-seed share <= 10%).
PER_SEED_SHARE = 0.10
UNSTABLE_RETIRE_COUNT = 2
GROUPS = ("D_E", "D_M", "D_H")
CLEAR_CACHE_FREQ = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_companion_sha256(path: Path) -> str:
    """Check a frozen artifact against its <name>.sha256 companion file."""
    actual = sha256_file(path)
    companion = path.with_name(path.name + ".sha256")
    if companion.exists():
        expected = companion.read_text().split()[0].strip()
        if actual != expected:
            raise RuntimeError(
                f"Frozen artifact {path} sha256 {actual} does not match its "
                f"companion lock {expected}; refusing to run."
            )
    return actual


def load_protocol_rules(protocol_path: Path) -> dict:
    protocol = read_json(protocol_path)
    section = protocol["t1c_data_sources_v1_1"]
    rules = {
        "attempt_caps": {
            group: int(cap)
            for group, cap in section["acquisition_rule"]["per_seed_attempt_cap"].items()
        },
        "n_target": int(section["targets_and_gates"]["N_target_per_source"]),
        "u_min": {
            group: int(value)
            for group, value in section["targets_and_gates"]["U_min"].items()
        },
        "protocol_revision": protocol["protocol_revision"],
    }
    for group in GROUPS:
        if group not in rules["attempt_caps"] or group not in rules["u_min"]:
            raise RuntimeError(f"protocol {protocol_path} missing rules for {group}")
    return rules


def build_pools(groups_payload: dict) -> dict:
    seeds = groups_payload["seeds"]
    tail = {int(seed) for seed in groups_payload["unsupported_tail_0_of_8"]}
    easy = sorted(int(seed) for seed, info in seeds.items() if info["group"] == "easy")
    medium = sorted(int(seed) for seed, info in seeds.items() if info["group"] == "medium")
    hard = sorted(
        int(seed)
        for seed, info in seeds.items()
        if info["group"] == "hard" and int(seed) not in tail
    )
    return {"D_E": easy[:30], "D_M": medium, "D_H": hard}


class SeedState:
    def __init__(self, env_seed: int):
        self.env_seed = env_seed
        self.attempts = 0
        self.successes = 0
        self.unstable_errors = 0

    def retired(self) -> bool:
        return self.unstable_errors >= UNSTABLE_RETIRE_COUNT

    def finished(self, attempt_cap: int, share_cap: int) -> bool:
        return (
            self.retired()
            or self.attempts >= attempt_cap
            or self.successes >= share_cap
        )


class GroupState:
    def __init__(self, group: str, pool, rules: dict):
        self.group = group
        self.pool = list(pool)
        self.attempt_cap = rules["attempt_caps"][group]
        self.n_target = rules["n_target"]
        self.u_min = rules["u_min"][group]
        self.share_cap = math.ceil(PER_SEED_SHARE * self.n_target)
        self.seeds = {seed: SeedState(seed) for seed in self.pool}

    def apply_row(self, row: dict) -> None:
        state = self.seeds[int(row["env_seed"])]
        state.attempts += 1
        if row.get("missingness_reason") == "UnStableError":
            state.unstable_errors += 1
        if row.get("success"):
            state.successes += 1

    def total_successes(self) -> int:
        return sum(state.successes for state in self.seeds.values())

    def unique_success_seeds(self) -> int:
        return sum(1 for state in self.seeds.values() if state.successes > 0)

    def n_eff(self):
        counts = [state.successes for state in self.seeds.values() if state.successes]
        if not counts:
            return None
        return (sum(counts) ** 2) / sum(count * count for count in counts)

    def target_met(self) -> bool:
        return (
            self.total_successes() >= self.n_target
            and self.unique_success_seeds() >= self.u_min
        )

    def all_finished(self) -> bool:
        return all(
            state.finished(self.attempt_cap, self.share_cap)
            for state in self.seeds.values()
        )

    def status(self) -> str:
        if self.target_met():
            return "complete"
        if self.all_finished():
            return "source_infeasible"
        return "collecting"

    def summary(self) -> dict:
        n_eff = self.n_eff()
        return {
            "group": self.group,
            "status": self.status(),
            "successful_trajectories": self.total_successes(),
            "n_target": self.n_target,
            "unique_success_seeds": self.unique_success_seeds(),
            "u_min": self.u_min,
            "n_eff": round(n_eff, 3) if n_eff is not None else None,
            "n_eff_gate_pass": (n_eff is not None and n_eff >= self.u_min),
            "attempts": sum(state.attempts for state in self.seeds.values()),
            "attempt_cap_per_seed": self.attempt_cap,
            "per_seed_share_cap": self.share_cap,
            "retired_seeds": sorted(
                state.env_seed for state in self.seeds.values() if state.retired()
            ),
            "per_seed": {
                str(seed): {
                    "attempts": state.attempts,
                    "successes": state.successes,
                    "unstable_errors": state.unstable_errors,
                }
                for seed, state in sorted(self.seeds.items())
            },
        }


def frozen_traversal(state: GroupState):
    """Yield (round, env_seed, attempt_index) per the frozen acquisition rule.

    The caller MUST apply each attempt's outcome to `state` before pulling the
    next item; the generator reads live state to decide skipping and stopping.
    """
    round_index = 0
    while True:
        any_attempted = False
        for env_seed in state.pool:
            if state.status() != "collecting":
                return
            seed_state = state.seeds[env_seed]
            if seed_state.finished(state.attempt_cap, state.share_cap):
                continue
            any_attempted = True
            yield round_index, env_seed, seed_state.attempts
        if not any_attempted:
            return
        round_index += 1


def manifest_path(run_dir: Path, group: str) -> Path:
    return run_dir / f"manifest_{group}.jsonl"


def state_path(run_dir: Path, group: str) -> Path:
    return run_dir / f"state_{group}.json"


def load_recorded_rows(run_dir: Path, group: str, pool, meta: dict) -> dict:
    """Load the append-only manifest into a {(env_seed, attempt_index): row}
    lookup, validating meta and the frozen policy-seed rule per row."""
    pool_set = set(pool)
    recorded = {}
    for index, row in enumerate(iter_jsonl(manifest_path(run_dir, group))):
        for key, expected in meta.items():
            if row.get(key) != expected:
                raise RuntimeError(
                    f"Refusing to resume {manifest_path(run_dir, group)} row "
                    f"{index}: {key} is {row.get(key)!r}, expected {expected!r}"
                )
        env_seed = int(row["env_seed"])
        attempt_index = int(row["attempt_index"])
        if env_seed not in pool_set:
            raise RuntimeError(
                f"Manifest row {index}: env_seed {env_seed} outside the frozen "
                f"{group} pool; refusing to resume."
            )
        if int(row["policy_seed"]) != POLICY_SEED_BASE + attempt_index:
            raise RuntimeError(
                f"Manifest row {index}: policy_seed {row['policy_seed']} != "
                f"{POLICY_SEED_BASE + attempt_index}"
            )
        key = (env_seed, attempt_index)
        if key in recorded:
            raise RuntimeError(f"Manifest row {index}: duplicate attempt {key}")
        recorded[key] = row
    return recorded


def load_dp_model(ckpt_path, action_dim, normalizer_zarr_path=None):
    import yaml
    from policy.DP.dp_model import DP

    config_path = repo_path(
        "policy", "DP", "diffusion_policy", "config", f"robot_dp_{action_dim}.yaml"
    )
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return DP(
        str(ckpt_path),
        n_obs_steps=cfg["n_obs_steps"],
        n_action_steps=cfg["n_action_steps"],
        normalizer_zarr_path=str(normalizer_zarr_path) if normalizer_zarr_path else None,
    )


def attempt_once(task_name, env_args, model, env_seed, policy_seed, tmp_dir: Path,
                 clear_cache: bool):
    import torch
    from envs.utils.create_actor import UnStableError
    from policy.DP.deploy_policy import encode_obs

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    run_args = dict(env_args)
    run_args.update(
        {
            "need_plan": False,
            "save_data": True,
            "save_path": str(tmp_dir),
            "render_freq": 0,
        }
    )
    if run_args.get("save_freq") is None:
        run_args["save_freq"] = 15

    env = make_task_env(task_name)
    try:
        if hasattr(model, "set_generator"):
            gen = torch.Generator(device="cuda:0")
            gen.manual_seed(int(policy_seed))
            model.set_generator(gen)
        model.reset_obs()
        try:
            env.setup_demo(now_ep_num=0, seed=env_seed, is_test=True, **run_args)
        except UnStableError as exc:
            return {
                "success": False,
                "steps": 0,
                "evaluated": False,
                "missingness_reason": "UnStableError",
                "missingness_detail": str(exc),
                "raw_hdf5": None,
                "raw_video": None,
            }
        env.set_instruction("t1c source collection")
        success = False
        while env.take_action_cnt < env.step_lim:
            observation = env.get_obs()
            obs = encode_obs(observation)
            actions = model.get_action(obs)
            for action in actions:
                env.take_action(action)
                observation = env.get_obs()
                obs = encode_obs(observation)
                model.update_obs(obs)
                if env.eval_success:
                    success = True
                    break
            if success:
                break
        raw_hdf5 = None
        raw_video = None
        if success:
            # The hdf5 is materialized here from the pkl cache that
            # take_action's control loop wrote at save_freq intervals;
            # failed attempts keep only their manifest row.
            env.merge_pkl_to_hdf5_video()
            raw_hdf5 = tmp_dir / "data" / "episode0.hdf5"
            raw_video = tmp_dir / "video" / "episode0.mp4"
        return {
            "success": bool(success),
            "steps": int(env.take_action_cnt),
            "evaluated": True,
            "missingness_reason": None,
            "missingness_detail": None,
            "raw_hdf5": raw_hdf5,
            "raw_video": raw_video,
        }
    finally:
        try:
            env.remove_data_cache()
        except Exception:
            pass
        try:
            env.close_env(clear_cache=clear_cache)
        except Exception:
            pass


def dry_run_attempt(groups_payload, env_seed, attempt_index):
    """Deterministic simulator for logic tests: success iff a stable hash draw
    falls under the seed's frozen posterior mean. No GPU, no files."""
    info = groups_payload["seeds"][str(env_seed)]
    draw_bytes = hashlib.sha256(f"t1c:{env_seed}:{attempt_index}".encode()).digest()
    draw = int.from_bytes(draw_bytes[:8], "big") / float(1 << 64)
    return {
        "success": draw < float(info["posterior_mean"]),
        "steps": 0,
        "evaluated": True,
        "missingness_reason": None,
        "missingness_detail": None,
        "raw_hdf5": None,
        "raw_video": None,
    }


def collect_group(args, rules, groups_payload, run_dir: Path):
    group = args.group
    pools = build_pools(groups_payload)
    state = GroupState(group, pools[group], rules)
    meta = {
        "task": args.task_name,
        "group": group,
        "protocol_revision": rules["protocol_revision"],
        "ckpt_path": str(args.ckpt_path),
    }
    if manifest_path(run_dir, group).exists() and not args.resume:
        raise RuntimeError(
            f"{manifest_path(run_dir, group)} already exists; pass --resume to "
            "continue it (manifests are append-only and never overwritten)."
        )
    recorded = load_recorded_rows(run_dir, group, state.pool, meta)
    if recorded:
        print(f"[{group}] resuming with {len(recorded)} recorded attempts")

    success_dir = run_dir / "successes" / group
    video_dir = run_dir / "videos" / group

    # Lazy: only pay env/model setup when the traversal actually goes live.
    runtime = {"model": None, "env_args": None, "live_attempts": 0}

    def run_live(env_seed, attempt_index, policy_seed):
        if args.dry_run:
            return dry_run_attempt(groups_payload, env_seed, attempt_index)
        if runtime["model"] is None:
            os.chdir(repo_path())
            runtime["env_args"] = load_task_args(args.task_name, args.task_config)
            runtime["model"] = load_dp_model(
                args.ckpt_path, args.action_dim,
                normalizer_zarr_path=args.normalizer_zarr,
            )
        runtime["live_attempts"] += 1
        tmp_dir = run_dir / ".tmp" / f"{group}_{env_seed}_a{attempt_index}"
        return attempt_once(
            args.task_name,
            runtime["env_args"],
            runtime["model"],
            env_seed,
            policy_seed,
            tmp_dir,
            clear_cache=(runtime["live_attempts"] % CLEAR_CACHE_FREQ == 0),
        )

    consumed = set()
    for round_index, env_seed, attempt_index in frozen_traversal(state):
        key = (env_seed, attempt_index)
        policy_seed = POLICY_SEED_BASE + attempt_index
        if key in recorded:
            row = recorded[key]
            if int(row["round"]) != round_index:
                raise RuntimeError(
                    f"[{group}] recorded attempt {key} has round {row['round']} "
                    f"but the frozen traversal reaches it in round {round_index}; "
                    "manifest is inconsistent with the acquisition rule."
                )
            if row.get("success"):
                hdf5 = row.get("hdf5_path")
                if not args.dry_run and (not hdf5 or not (run_dir / hdf5).is_file()):
                    raise RuntimeError(
                        f"[{group}] recorded success {key} missing on disk "
                        f"({hdf5!r}); refusing to silently recollect."
                    )
            consumed.add(key)
            state.apply_row(row)
            continue

        started = time.time()
        result = run_live(env_seed, attempt_index, policy_seed)
        hdf5_rel = None
        video_rel = None
        if result["success"] and result["raw_hdf5"] is not None:
            if not Path(result["raw_hdf5"]).is_file():
                raise RuntimeError(
                    f"success reported but no hdf5 at {result['raw_hdf5']}"
                )
            success_dir.mkdir(parents=True, exist_ok=True)
            dest = success_dir / f"episode_{env_seed}_a{attempt_index}.hdf5"
            shutil.move(str(result["raw_hdf5"]), str(dest))
            hdf5_rel = str(dest.relative_to(run_dir))
            if result["raw_video"] and Path(result["raw_video"]).is_file():
                video_dir.mkdir(parents=True, exist_ok=True)
                vdest = video_dir / f"episode_{env_seed}_a{attempt_index}.mp4"
                shutil.move(str(result["raw_video"]), str(vdest))
                video_rel = str(vdest.relative_to(run_dir))
        if not args.dry_run:
            shutil.rmtree(
                run_dir / ".tmp" / f"{group}_{env_seed}_a{attempt_index}",
                ignore_errors=True,
            )
        row = {
            **meta,
            "round": round_index,
            "env_seed": int(env_seed),
            "attempt_index": int(attempt_index),
            "policy_seed": int(policy_seed),
            "evaluated": result["evaluated"],
            "success": result["success"],
            "steps": result["steps"],
            "missingness_reason": result["missingness_reason"],
            "missingness_detail": result["missingness_detail"],
            "hdf5_path": hdf5_rel,
            "video_path": video_rel,
            "wall_seconds": round(time.time() - started, 2),
        }
        append_jsonl(manifest_path(run_dir, group), row)
        state.apply_row(row)
        write_json_atomic(
            state_path(run_dir, group),
            {**meta, "policy_seed_base": POLICY_SEED_BASE, **state.summary()},
        )
        tag = (
            "UNSTABLE"
            if row["missingness_reason"] == "UnStableError"
            else ("success" if row["success"] else "fail")
        )
        print(
            f"[{group}] r{round_index} seed={env_seed} a{attempt_index} {tag} "
            f"({state.total_successes()}/{state.n_target}, "
            f"U={state.unique_success_seeds()}/{state.u_min})"
        )

    leftover = set(recorded) - consumed
    if leftover:
        raise RuntimeError(
            f"[{group}] {len(leftover)} recorded attempts were never reached by "
            f"the frozen traversal (e.g. {sorted(leftover)[:5]}); manifest is "
            "inconsistent with the acquisition rule."
        )

    final = {**meta, "policy_seed_base": POLICY_SEED_BASE, **state.summary()}
    write_json_atomic(state_path(run_dir, group), final)
    print(
        f"[{group}] final status: {final['status']} "
        f"({final['successful_trajectories']}/{final['n_target']} trajectories, "
        f"U={final['unique_success_seeds']}, n_eff={final['n_eff']})"
    )
    if final["status"] == "source_infeasible":
        print(
            f"[{group}] SOURCE-INFEASIBLE under the frozen stop rule: caps "
            "exhausted before N_target AND U_min. Per protocol.t1.v1.1 this "
            "halts T1d for the task (no seed swapping, no cap raising)."
        )
    return final


def write_report(args, rules, run_dir: Path, provenance: dict):
    groups = {}
    for group in GROUPS:
        path = state_path(run_dir, group)
        if not path.exists():
            raise RuntimeError(f"missing {path}; run all three groups first")
        groups[group] = read_json(path)
    statuses = {group: payload["status"] for group, payload in groups.items()}
    all_complete = all(status == "complete" for status in statuses.values())
    n_eff_pass = all(payload["n_eff_gate_pass"] for payload in groups.values())
    report = {
        "record": f"capability_transport.t1c_collection.{args.task_name}.v1",
        "protocol_revision": rules["protocol_revision"],
        "task": args.task_name,
        "provenance": provenance,
        "groups": {
            group: {
                key: payload[key]
                for key in (
                    "status",
                    "successful_trajectories",
                    "n_target",
                    "unique_success_seeds",
                    "u_min",
                    "n_eff",
                    "n_eff_gate_pass",
                    "attempts",
                    "retired_seeds",
                )
            }
            for group, payload in groups.items()
        },
        "verdict": (
            "PASS — all E/M/H sources complete and n_eff gates pass; T1d may "
            "proceed once D_Q is collected"
            if all_complete and n_eff_pass
            else "FAIL — see group statuses; per the frozen stop rule T1d halts "
            "for this task"
        ),
    }
    report_path = run_dir / "t1c_collection_report.json"
    write_json_atomic(report_path, report)

    total_attempts = sum(payload["attempts"] for payload in groups.values())
    total_stored = sum(
        payload["successful_trajectories"] for payload in groups.values()
    )
    gpu_hours = 0.0
    for group in GROUPS:
        for row in iter_jsonl(manifest_path(run_dir, group)):
            gpu_hours += float(row.get("wall_seconds") or 0.0) / 3600.0
    if args.dry_run:
        print("dry run: NOT appending to the shared efficiency ledger")
    else:
        append_jsonl(
            repo_path("experiments", "capability_transport", "efficiency_ledger.jsonl"),
            {
                "stage": "t1c_collection",
                "task": args.task_name,
                "episodes_simulated": total_attempts,
                "successful_trajectories_stored": total_stored,
                "expert_demos_used": 0,
                "training_runs": 0,
                "gradient_steps": 0,
                "gpu_hours": round(gpu_hours, 3),
                "wall_clock": "see per-attempt wall_seconds in run manifests",
                "run_dir": str(run_dir),
            },
        )
    print(f"Wrote {report_path}")
    print(f"Verdict: {report['verdict']}")


def main():
    parser = argparse.ArgumentParser(
        description="T1c round-robin success-first source collection (protocol.t1.v1.1)."
    )
    parser.add_argument("--task", dest="task_name", required=True)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--normalizer-zarr", type=Path)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=repo_path("experiments", "capability_transport", "protocol.t1.v1.1.json"),
    )
    parser.add_argument("--groups-file", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate outcomes deterministically from frozen posteriors (no GPU).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Merge the three group states into t1c_collection_report.json and "
        "append the efficiency ledger entry.",
    )
    args = parser.parse_args()

    groups_file = args.groups_file or repo_path(
        "experiments",
        "capability_transport",
        f"difficulty_groups.{args.task_name}.v1.json",
    )
    groups_sha = verify_companion_sha256(groups_file)
    protocol_sha = verify_companion_sha256(args.protocol)
    rules = load_protocol_rules(args.protocol)
    groups_payload = read_json(groups_file)
    if groups_payload["task"] != args.task_name:
        raise RuntimeError(
            f"groups file task {groups_payload['task']!r} != --task {args.task_name!r}"
        )

    args.run_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "groups_file": str(groups_file),
        "groups_file_sha256": groups_sha,
        "protocol_file": str(args.protocol),
        "protocol_sha256": protocol_sha,
        "ckpt_path": str(args.ckpt_path),
        "ckpt_sha256": sha256_file(args.ckpt_path) if args.ckpt_path.is_file() else None,
        "policy_seed_base": POLICY_SEED_BASE,
    }
    write_json_atomic(args.run_dir / "provenance.json", provenance)

    if args.report:
        write_report(args, rules, args.run_dir, provenance)
        return
    if not args.group:
        parser.error("--group is required unless --report is set")
    if not args.dry_run and not args.ckpt_path.is_file():
        raise RuntimeError(f"missing checkpoint: {args.ckpt_path}")
    collect_group(args, rules, groups_payload, args.run_dir)


if __name__ == "__main__":
    main()
