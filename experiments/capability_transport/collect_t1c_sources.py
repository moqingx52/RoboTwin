#!/usr/bin/env python3
"""T1c source acquisition: round-robin success-first collection with pi0.

Implements protocol.t1.v1.2.json (which amends v1.1's D_H rules only) for
tasks that passed the source-feasibility gate (t1c_source_feasibility.v1.json).

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
    trajectories >= per-seed success cap; retired after 2 UnStableErrors.
  - Store ONLY complete successful trajectories (hdf5 via
    merge_pkl_to_hdf5_video); every attempt (success, failure, UnStableError)
    is appended to the group manifest and never deleted.
  - Stop, mode "target" (D_E, D_M): N_target successes AND U_h >= U_min ->
    complete; every candidate seed finished/retired first -> source_infeasible.
  - Stop, mode "fixed_budget" (D_H, v1.2): NO N_target; collection runs until
    the frozen opportunity set is exhausted (budget_exhausted). Admissibility
    is then judged in --report by the four-part joint gate
    G_H = [U_H>=10] AND [n_eff_H>=10] AND [M_H>=M_H_min] AND
    [all frozen dose points satisfy w = rho/q <= 3], with
    q_h = M_h / (M_E + M_M + M_H) (Base200 and D_Q excluded).
    G_H=0 -> source-infeasibility certificate, T1d halts.

Dry-run modes:
  --dry-run          posterior-MEAN hash draws. Logic/resume SMOKE TEST only:
                     1/8 hard seeds have posterior mean 0.2 vs empirical 0.125,
                     so its yields are NOT predictive of real collection.
  --mc-feasibility   posterior-predictive Monte Carlo for D_H: per replicate
                     draw p_s ~ Beta(1+s, 1+R-s) per seed, simulate the frozen
                     acquisition, report per-part and joint G_H pass rates.

Resume is exact: recorded manifest rows are replayed THROUGH the frozen
traversal simulation, and live collection begins at the first attempt the
traversal requests that has no recorded outcome. The resumed realization is
therefore identical to an uninterrupted serial run.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
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

# Frozen in protocol.t1.v1.2.json collector_rules:
# offset 6000 is disjoint from T1b 5000..5007, legacy census 3000+, phase1 0+.
POLICY_SEED_BASE = 6000
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
    """Read the machine-readable collector_rules section of protocol.t1.v1.2."""
    protocol = read_json(protocol_path)
    section = protocol["collector_rules"]
    if int(section["policy_seed_base"]) != POLICY_SEED_BASE:
        raise RuntimeError(
            f"protocol policy_seed_base {section['policy_seed_base']} != "
            f"code constant {POLICY_SEED_BASE}"
        )
    if int(section["unstable_retire_count"]) != UNSTABLE_RETIRE_COUNT:
        raise RuntimeError("protocol unstable_retire_count mismatch")
    groups = {}
    for group in GROUPS:
        if group not in section["groups"]:
            raise RuntimeError(f"protocol {protocol_path} missing rules for {group}")
        spec = section["groups"][group]
        if spec["mode"] not in ("target", "fixed_budget"):
            raise RuntimeError(f"{group}: unknown mode {spec['mode']!r}")
        if spec["mode"] == "target" and spec["n_target"] is None:
            raise RuntimeError(f"{group}: target mode requires n_target")
        groups[group] = {
            "mode": spec["mode"],
            "n_target": None if spec["n_target"] is None else int(spec["n_target"]),
            "u_min": int(spec["u_min"]),
            "attempt_cap": int(spec["attempt_cap"]),
            "success_cap": int(spec["success_cap"]),
        }
    gate = section["joint_gate_G_H"]
    return {
        "groups": groups,
        "joint_gate_G_H": {
            "u_min_H": int(gate["u_min_H"]),
            "n_eff_min_H": float(gate["n_eff_min_H"]),
            "m_h_min": int(gate["m_h_min"]),
            "w_max": float(gate["w_max"]),
            "rho_h_max_D_H": float(gate["rho_h_max"]["D_H"]),
            "q_definition": gate["q_definition"],
            "frozen_dose_points": {
                name: {g: float(v) for g, v in point.items()}
                for name, point in gate["frozen_dose_points"].items()
            },
        },
        "mc_seed": int(section["mc_seed"]),
        "mc_replicates_default": int(section["mc_replicates_default"]),
        "protocol_revision": protocol["protocol_revision"],
    }


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

    def finished(self, attempt_cap: int, success_cap: int) -> bool:
        return (
            self.retired()
            or self.attempts >= attempt_cap
            or self.successes >= success_cap
        )


class GroupState:
    def __init__(self, group: str, pool, rules: dict):
        spec = rules["groups"][group]
        self.group = group
        self.pool = list(pool)
        self.mode = spec["mode"]
        self.attempt_cap = spec["attempt_cap"]
        self.n_target = spec["n_target"]
        self.u_min = spec["u_min"]
        self.success_cap = spec["success_cap"]
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
            self.mode == "target"
            and self.total_successes() >= self.n_target
            and self.unique_success_seeds() >= self.u_min
        )

    def all_finished(self) -> bool:
        return all(
            state.finished(self.attempt_cap, self.success_cap)
            for state in self.seeds.values()
        )

    def status(self) -> str:
        if self.mode == "target":
            if self.target_met():
                return "complete"
            if self.all_finished():
                return "source_infeasible"
            return "collecting"
        # fixed_budget (D_H, v1.2): no target; run the frozen opportunity set
        # dry. Admissibility is judged by the joint gate G_H in --report.
        if self.all_finished():
            return "budget_exhausted"
        return "collecting"

    def summary(self) -> dict:
        n_eff = self.n_eff()
        return {
            "group": self.group,
            "mode": self.mode,
            "status": self.status(),
            "successful_trajectories": self.total_successes(),
            "n_target": self.n_target,
            "unique_success_seeds": self.unique_success_seeds(),
            "u_min": self.u_min,
            "n_eff": round(n_eff, 3) if n_eff is not None else None,
            "n_eff_gate_pass": (n_eff is not None and n_eff >= self.u_min),
            "attempts": sum(state.attempts for state in self.seeds.values()),
            "attempt_cap_per_seed": self.attempt_cap,
            "per_seed_success_cap": self.success_cap,
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
            if seed_state.finished(state.attempt_cap, state.success_cap):
                continue
            any_attempted = True
            yield round_index, env_seed, seed_state.attempts
        if not any_attempted:
            return
        round_index += 1


def manifest_path(run_dir: Path, group: str, seed: int | None = None) -> Path:
    if seed is not None:
        return run_dir / f"manifest_{group}.seed_{int(seed)}.jsonl"
    return run_dir / f"manifest_{group}.jsonl"


def state_path(run_dir: Path, group: str, seed: int | None = None) -> Path:
    if seed is not None:
        return run_dir / f"state_{group}.seed_{int(seed)}.json"
    return run_dir / f"state_{group}.json"


def load_recorded_rows(path: Path, pool, meta: dict) -> dict:
    """Load the append-only manifest into a {(env_seed, attempt_index): row}
    lookup, validating meta and the frozen policy-seed rule per row."""
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
        attempt_index = int(row["attempt_index"])
        if env_seed not in pool_set:
            raise RuntimeError(
                f"Manifest row {index} in {path}: env_seed {env_seed} outside "
                f"the frozen pool; refusing to resume."
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
    """Posterior-MEAN smoke test: success iff a stable hash draw falls under
    the seed's frozen posterior mean. Logic/resume testing ONLY — for 1/8
    hard seeds the posterior mean (0.2) overstates the empirical rate (0.125),
    so smoke-test yields are NOT forecasts of real collection. Predictive
    simulation is --mc-feasibility. No GPU, no files."""
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


def mc_feasibility(args, rules, groups_payload, provenance: dict):
    """Posterior-predictive Monte Carlo for the D_H joint gate (v1.2).

    Per replicate: draw p_s ~ Beta(1 + s, 1 + R - s) for every supported-hard
    seed from the frozen census counts, simulate the frozen acquisition (up to
    attempt_cap attempts per seed, seed stops at success_cap successes), then
    evaluate all four parts of G_H assuming D_E/D_M meet their targets
    (M_E = M_M = 90). An empirical fixed-p variant (p_s = s/R) is reported
    alongside. With no global stop rule in fixed_budget mode, seeds are
    independent, so per-seed simulation is exactly the round-robin outcome.
    """
    import random

    spec = rules["groups"]["D_H"]
    if spec["mode"] != "fixed_budget":
        raise RuntimeError("--mc-feasibility requires D_H in fixed_budget mode")
    gate_rules = rules["joint_gate_G_H"]
    pool = build_pools(groups_payload)["D_H"]
    seeds_info = groups_payload["seeds"]
    census = []
    for env_seed in pool:
        info = seeds_info[str(env_seed)]
        s = int(info["successes"])
        r = int(info["evaluated_repeats"])
        census.append((env_seed, s, r))

    replicates = args.mc_replicates or rules["mc_replicates_default"]
    m_e = rules["groups"]["D_E"]["n_target"]
    m_m = rules["groups"]["D_M"]["n_target"]
    rng = random.Random(rules["mc_seed"])

    def simulate(draw_p):
        counts = {"joint": 0, "1_coverage": 0, "2_ess": 0, "3_volume": 0,
                  "4_dose": 0}
        m_h_values = []
        for _ in range(replicates):
            m_s = []
            for env_seed, s, r in census:
                p = draw_p(s, r)
                successes = 0
                for _attempt in range(spec["attempt_cap"]):
                    if successes >= spec["success_cap"]:
                        break
                    if rng.random() < p:
                        successes += 1
                m_s.append(successes)
            m_h = sum(m_s)
            u_h = sum(1 for c in m_s if c > 0)
            sq = sum(c * c for c in m_s)
            n_eff = (m_h * m_h / sq) if sq else None
            gate = evaluate_joint_gate(
                gate_rules, {"D_E": m_e, "D_M": m_m, "D_H": m_h}, u_h, n_eff
            )
            m_h_values.append(m_h)
            counts["joint"] += gate["G_H"]
            counts["1_coverage"] += gate["parts"]["1_coverage_U_H"]["pass"]
            counts["2_ess"] += gate["parts"]["2_ess_n_eff_H"]["pass"]
            counts["3_volume"] += gate["parts"]["3_volume_M_H"]["pass"]
            counts["4_dose"] += gate["parts"]["4_dose_feasibility"]["pass"]
        m_h_values.sort()
        return {
            "pass_rates": {k: round(v / replicates, 4) for k, v in counts.items()},
            "M_H": {
                "mean": round(sum(m_h_values) / replicates, 2),
                "p05": m_h_values[int(0.05 * replicates)],
                "p50": m_h_values[int(0.50 * replicates)],
                "p95": m_h_values[int(0.95 * replicates)],
            },
        }

    posterior_predictive = simulate(
        lambda s, r: rng.betavariate(1 + s, 1 + r - s)
    )
    empirical = simulate(lambda s, r: s / r)

    record = {
        "record": f"capability_transport.t1c_dh_mc_feasibility.{args.task_name}.v1",
        "protocol_revision": rules["protocol_revision"],
        "task": args.task_name,
        "mc_seed": rules["mc_seed"],
        "replicates": replicates,
        "assumptions": {
            "M_E": m_e,
            "M_M": m_m,
            "note": "parts 3-4 of G_H assume D_E/D_M reach their targets; "
            "D_H seeds simulated independently (valid: fixed_budget has no "
            "global stop).",
        },
        "d_h_pool": [seed for seed, _, _ in census],
        "posterior_predictive_p_s_beta": posterior_predictive,
        "empirical_fixed_p_hat": empirical,
        "provenance": provenance,
    }
    out_path = args.mc_output or repo_path(
        "experiments", "capability_transport",
        f"t1c_dh_mc_feasibility.{args.task_name}.v1.json",
    )
    write_json_atomic(Path(out_path), record)
    print(f"Wrote {out_path}")
    print(
        "posterior predictive: joint G_H pass "
        f"{posterior_predictive['pass_rates']['joint']:.1%}, "
        f"M_H mean {posterior_predictive['M_H']['mean']}"
    )
    print(
        "empirical p_hat:      joint G_H pass "
        f"{empirical['pass_rates']['joint']:.1%}, "
        f"M_H mean {empirical['M_H']['mean']}"
    )


def _materialize_success(run_dir: Path, group: str, env_seed: int, attempt_index: int,
                         result: dict):
    hdf5_rel = None
    video_rel = None
    if result["success"] and result["raw_hdf5"] is not None:
        if not Path(result["raw_hdf5"]).is_file():
            raise RuntimeError(f"success reported but no hdf5 at {result['raw_hdf5']}")
        success_dir = run_dir / "successes" / group
        success_dir.mkdir(parents=True, exist_ok=True)
        dest = success_dir / f"episode_{env_seed}_a{attempt_index}.hdf5"
        shutil.move(str(result["raw_hdf5"]), str(dest))
        hdf5_rel = str(dest.relative_to(run_dir))
        if result["raw_video"] and Path(result["raw_video"]).is_file():
            video_dir = run_dir / "videos" / group
            video_dir.mkdir(parents=True, exist_ok=True)
            vdest = video_dir / f"episode_{env_seed}_a{attempt_index}.mp4"
            shutil.move(str(result["raw_video"]), str(vdest))
            video_rel = str(vdest.relative_to(run_dir))
    shutil.rmtree(
        run_dir / ".tmp" / f"{group}_{env_seed}_a{attempt_index}",
        ignore_errors=True,
    )
    return hdf5_rel, video_rel


def replay_canonical(run_dir: Path, group: str, pool, rules, meta: dict):
    state = GroupState(group, pool, rules)
    recorded = load_recorded_rows(manifest_path(run_dir, group), pool, meta)
    for round_index, env_seed, attempt_index in frozen_traversal(state):
        key = (env_seed, attempt_index)
        if key not in recorded:
            break
        row = recorded[key]
        if int(row["round"]) != round_index:
            raise RuntimeError(
                f"[{group}] canonical {key} round {row['round']} != traversal {round_index}"
            )
        state.apply_row(row)
    return state, recorded


def current_round_pending(state: GroupState):
    """Seeds that share the current round-robin wave (same attempt_index)."""
    if state.status() != "collecting":
        return None, []
    unfinished = [
        seed for seed in state.pool
        if not state.seeds[seed].finished(state.attempt_cap, state.success_cap)
    ]
    if not unfinished:
        return None, []
    attempt_index = min(state.seeds[seed].attempts for seed in unfinished)
    pending = [
        seed for seed in state.pool
        if not state.seeds[seed].finished(state.attempt_cap, state.success_cap)
        and state.seeds[seed].attempts == attempt_index
    ]
    return attempt_index, pending


def run_one_attempt(args, rules, groups_payload, run_dir: Path):
    """Execute a single specified (seed, attempt_index). Used by round-parallel D_M."""
    group = args.group
    env_seed = int(args.env_seed)
    attempt_index = int(args.attempt_index)
    round_index = int(args.round_index)
    pools = build_pools(groups_payload)
    if env_seed not in pools[group]:
        raise RuntimeError(f"env_seed {env_seed} not in frozen {group} pool")
    meta = {
        "task": args.task_name,
        "group": group,
        "protocol_revision": rules["protocol_revision"],
        "ckpt_path": str(args.ckpt_path),
    }
    man_path = manifest_path(run_dir, group, seed=env_seed)
    recorded = load_recorded_rows(man_path, [env_seed], meta)
    key = (env_seed, attempt_index)
    if key in recorded:
        print(f"[{group}] seed={env_seed} a{attempt_index} already in shard, skip")
        return
    policy_seed = POLICY_SEED_BASE + attempt_index
    os.chdir(repo_path())
    env_args = load_task_args(args.task_name, args.task_config)
    model = load_dp_model(
        args.ckpt_path, args.action_dim,
        normalizer_zarr_path=args.normalizer_zarr,
    )
    tmp_dir = run_dir / ".tmp" / f"{group}_{env_seed}_a{attempt_index}"
    started = time.time()
    result = attempt_once(
        args.task_name, env_args, model, env_seed, policy_seed, tmp_dir, clear_cache=True,
    )
    hdf5_rel, video_rel = _materialize_success(run_dir, group, env_seed, attempt_index, result)
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
    append_jsonl(man_path, row)
    tag = (
        "UNSTABLE"
        if row["missingness_reason"] == "UnStableError"
        else ("success" if row["success"] else "fail")
    )
    print(f"[{group}] r{round_index} seed={env_seed} a{attempt_index} {tag}")


def parallel_target_rounds(args, rules, groups_payload, run_dir: Path):
    """Round-parallel D_E/D_M: run one round-robin wave concurrently, commit in
    frozen seed order, stop committing once N_target AND U_min are met.

    Same official manifest as serial earliest-stop; extra in-flight attempts
    after the stopping seed are not committed.
    """
    group = args.group
    if rules["groups"][group]["mode"] != "target":
        raise RuntimeError("--parallel-rounds is for target-mode groups (D_E/D_M)")
    gpu_ids = [int(x) for x in str(args.gpu_ids).split()]
    workers_per_gpu = int(args.workers_per_gpu)
    max_workers = len(gpu_ids) * workers_per_gpu
    pool = build_pools(groups_payload)[group]
    meta = {
        "task": args.task_name,
        "group": group,
        "protocol_revision": rules["protocol_revision"],
        "ckpt_path": str(args.ckpt_path),
    }
    canonical = manifest_path(run_dir, group)
    script = str(Path(__file__).resolve())
    python = sys.executable

    while True:
        state, recorded = replay_canonical(run_dir, group, pool, rules, meta)
        write_json_atomic(
            state_path(run_dir, group),
            {**meta, "policy_seed_base": POLICY_SEED_BASE, **state.summary()},
        )
        if state.status() != "collecting":
            print(
                f"[{group}] parallel-rounds done: {state.status()} "
                f"M={state.total_successes()} U={state.unique_success_seeds()} "
                f"n_eff={state.n_eff()}"
            )
            return
        round_index, pending = current_round_pending(state)
        print(
            f"[{group}] wave r{round_index}: {len(pending)} seeds, "
            f"M={state.total_successes()}/{state.n_target}, "
            f"U={state.unique_success_seeds()}/{state.u_min}, "
            f"workers<={max_workers}"
        )
        for wave_start in range(0, len(pending), max_workers):
            state, recorded = replay_canonical(run_dir, group, pool, rules, meta)
            if state.target_met():
                break
            wave = pending[wave_start:wave_start + max_workers]
            procs = []
            for i, seed in enumerate(wave):
                gpu = gpu_ids[i // workers_per_gpu]
                log_path = run_dir / f"{group}.seed_{seed}.a{round_index}.log"
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["OMP_NUM_THREADS"] = "1"
                env["MKL_NUM_THREADS"] = "1"
                env["PYTHONUNBUFFERED"] = "1"
                cmd = [
                    python, script,
                    "--task", args.task_name,
                    "--task-config", args.task_config,
                    "--group", group,
                    "--ckpt-path", str(args.ckpt_path),
                    "--run-dir", str(run_dir),
                    "--run-attempt",
                    "--env-seed", str(seed),
                    "--attempt-index", str(round_index),
                    "--round-index", str(round_index),
                ]
                if args.normalizer_zarr:
                    cmd.extend(["--normalizer-zarr", str(args.normalizer_zarr)])
                log_f = open(log_path, "a", encoding="utf-8")
                proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
                procs.append((proc, log_f, seed, gpu))
                print(f"  GPU {gpu}: {group} seed={seed} a{round_index}")
            fail = []
            for proc, log_f, seed, gpu in procs:
                rc = proc.wait()
                log_f.close()
                if rc != 0:
                    fail.append((seed, rc))
            if fail:
                raise RuntimeError(f"[{group}] workers failed: {fail}; re-run to resume")
            state, recorded = replay_canonical(run_dir, group, pool, rules, meta)
            for seed in pending:
                if state.target_met():
                    print(
                        f"[{group}] target met; not committing remaining r{round_index} "
                        f"seed={seed}+ (matches serial earliest-stop)"
                    )
                    break
                key = (seed, round_index)
                if key in recorded:
                    continue
                shard = load_recorded_rows(
                    manifest_path(run_dir, group, seed=seed), [seed], meta
                )
                if key not in shard:
                    continue
                row = shard[key]
                append_jsonl(canonical, row)
                state.apply_row(row)
                recorded[key] = row
                write_json_atomic(
                    state_path(run_dir, group),
                    {**meta, "policy_seed_base": POLICY_SEED_BASE, **state.summary()},
                )
                tag = "success" if row["success"] else "fail"
                print(
                    f"[{group}] commit r{round_index} seed={seed} a{round_index} {tag} "
                    f"({state.total_successes()}/{state.n_target}, "
                    f"U={state.unique_success_seeds()}/{state.u_min})"
                )
            if state.target_met():
                break


def collect_group(args, rules, groups_payload, run_dir: Path):
    group = args.group
    pools = build_pools(groups_payload)
    frozen_pool = list(pools[group])
    shard_seed = None
    if args.only_seeds:
        if group != "D_H":
            raise RuntimeError(
                "--only-seeds is only protocol-legal for D_H (fixed_budget, "
                "seeds independent). D_E/D_M must stay serial because the "
                "stop condition is evaluated before every attempt."
            )
        requested = [int(seed) for seed in args.only_seeds]
        unknown = [seed for seed in requested if seed not in frozen_pool]
        if unknown:
            raise RuntimeError(
                f"--only-seeds {unknown} are not in the frozen D_H pool {frozen_pool}"
            )
        if len(requested) != 1:
            raise RuntimeError("--only-seeds currently accepts exactly one seed per worker")
        shard_seed = requested[0]
        pool = [shard_seed]
    else:
        pool = frozen_pool

    state = GroupState(group, pool, rules)
    meta = {
        "task": args.task_name,
        "group": group,
        "protocol_revision": rules["protocol_revision"],
        "ckpt_path": str(args.ckpt_path),
    }
    man_path = manifest_path(run_dir, group, seed=shard_seed)
    st_path = state_path(run_dir, group, seed=shard_seed)
    if man_path.exists() and not args.resume:
        raise RuntimeError(
            f"{man_path} already exists; pass --resume to "
            "continue it (manifests are append-only and never overwritten)."
        )
    recorded = load_recorded_rows(man_path, state.pool, meta)
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
        append_jsonl(man_path, row)
        state.apply_row(row)
        write_json_atomic(
            st_path,
            {**meta, "policy_seed_base": POLICY_SEED_BASE, **state.summary()},
        )
        tag = (
            "UNSTABLE"
            if row["missingness_reason"] == "UnStableError"
            else ("success" if row["success"] else "fail")
        )
        target_str = (
            f"{state.total_successes()}/{state.n_target}"
            if state.mode == "target"
            else f"{state.total_successes()} (fixed budget)"
        )
        print(
            f"[{group}] r{round_index} seed={env_seed} a{attempt_index} {tag} "
            f"({target_str}, U={state.unique_success_seeds()}/{state.u_min})"
        )

    leftover = set(recorded) - consumed
    if leftover:
        raise RuntimeError(
            f"[{group}] {len(leftover)} recorded attempts were never reached by "
            f"the frozen traversal (e.g. {sorted(leftover)[:5]}); manifest is "
            "inconsistent with the acquisition rule."
        )

    final = {**meta, "policy_seed_base": POLICY_SEED_BASE, **state.summary()}
    write_json_atomic(st_path, final)
    print(
        f"[{group}] final status: {final['status']} "
        f"({final['successful_trajectories']} trajectories, "
        f"U={final['unique_success_seeds']}, n_eff={final['n_eff']})"
    )
    if final["status"] == "source_infeasible":
        print(
            f"[{group}] SOURCE-INFEASIBLE under the frozen stop rule: caps "
            "exhausted before N_target AND U_min. Per protocol.t1.v1.2 this "
            "halts T1d for the task (no seed swapping, no cap raising)."
        )
    if final["status"] == "budget_exhausted":
        print(
            f"[{group}] fixed opportunity budget exhausted; admissibility is "
            "judged by the joint gate G_H in --report (protocol.t1.v1.2)."
        )
    return final


def evaluate_joint_gate(gate_rules: dict, m: dict, u_h: int, n_eff_h) -> dict:
    """Four-part joint gate G_H (protocol.t1.v1.2). Needs all three realized
    volumes M_E, M_M, M_H because parts 3-4 use q_h = M_h/(M_E+M_M+M_H)."""
    m_total = m["D_E"] + m["D_M"] + m["D_H"]
    q = {g: (m[g] / m_total if m_total else 0.0) for g in GROUPS}
    w_max = gate_rules["w_max"]

    part_1 = u_h >= gate_rules["u_min_H"]
    part_2 = n_eff_h is not None and n_eff_h >= gate_rules["n_eff_min_H"]
    part_3 = m["D_H"] >= gate_rules["m_h_min"]

    # Part 4a: ceiling check at the frozen max hard dose rho_H_max = 0.5.
    ceiling_ok = (
        q["D_H"] > 0 and gate_rules["rho_h_max_D_H"] / q["D_H"] <= w_max
    )
    # Part 4b: every numerically frozen dose point satisfies w_h <= 3 for all
    # components with rho_h > 0, under the frozen q definition.
    dose_points = {}
    for name, point in gate_rules["frozen_dose_points"].items():
        w_point = {}
        ok = True
        for g in GROUPS:
            rho = point.get(g, 0.0)
            if rho <= 0:
                continue
            w = (rho / q[g]) if q[g] > 0 else float("inf")
            w_point[g] = round(w, 4) if w != float("inf") else "inf"
            ok = ok and w <= w_max
        dose_points[name] = {"w": w_point, "pass": ok}
    part_4 = ceiling_ok and all(p["pass"] for p in dose_points.values())

    return {
        "q_definition": gate_rules["q_definition"],
        "q": {g: round(q[g], 6) for g in GROUPS},
        "parts": {
            "1_coverage_U_H": {"value": u_h, "min": gate_rules["u_min_H"], "pass": part_1},
            "2_ess_n_eff_H": {
                "value": round(n_eff_h, 3) if n_eff_h is not None else None,
                "min": gate_rules["n_eff_min_H"],
                "pass": part_2,
            },
            "3_volume_M_H": {"value": m["D_H"], "min": gate_rules["m_h_min"], "pass": part_3},
            "4_dose_feasibility": {
                "rho_H_max": gate_rules["rho_h_max_D_H"],
                "w_max": w_max,
                "ceiling_w_H_at_rho_H_max": (
                    round(gate_rules["rho_h_max_D_H"] / q["D_H"], 4)
                    if q["D_H"] > 0
                    else "inf"
                ),
                "frozen_dose_points": dose_points,
                "pass": part_4,
            },
        },
        "G_H": part_1 and part_2 and part_3 and part_4,
    }


def merge_dh_shards(args, rules, groups_payload, run_dir: Path) -> dict:
    """Concatenate per-seed D_H manifests into the canonical group files.

    Seed-parallel workers are protocol-equivalent to serial D_H: fixed_budget
    has no global earliest-stop, and round r for seed s attempt k is k in both
    the 1-seed shard traversal and the 15-seed group traversal.
    """
    pool = build_pools(groups_payload)["D_H"]
    meta = {
        "task": args.task_name,
        "group": "D_H",
        "protocol_revision": rules["protocol_revision"],
        "ckpt_path": str(args.ckpt_path),
    }
    recorded = {}
    missing = []
    for seed in pool:
        path = manifest_path(run_dir, "D_H", seed=seed)
        if not path.exists():
            missing.append(seed)
            continue
        recorded.update(load_recorded_rows(path, [seed], meta))
    if missing:
        raise RuntimeError(
            f"D_H seed shards missing for {missing}; refusing to merge."
        )
    state = GroupState("D_H", pool, rules)
    ordered = []
    consumed = set()
    for round_index, env_seed, attempt_index in frozen_traversal(state):
        key = (env_seed, attempt_index)
        if key not in recorded:
            raise RuntimeError(
                f"D_H merge: traversal requested {key} but no shard row exists"
            )
        row = recorded[key]
        if int(row["round"]) != round_index:
            raise RuntimeError(
                f"D_H merge: shard row {key} has round {row['round']}, "
                f"traversal round {round_index}"
            )
        ordered.append(row)
        consumed.add(key)
        state.apply_row(row)
    leftover = set(recorded) - consumed
    if leftover:
        raise RuntimeError(
            f"D_H merge leftover shard rows: {sorted(leftover)[:5]}"
        )
    if state.status() != "budget_exhausted":
        raise RuntimeError(
            f"D_H merge status {state.status()!r}, expected budget_exhausted"
        )
    merged_path = manifest_path(run_dir, "D_H")
    tmp = merged_path.with_suffix(".jsonl.tmp")
    if tmp.exists():
        tmp.unlink()
    for row in ordered:
        append_jsonl(tmp, row)
    tmp.replace(merged_path)
    final = {**meta, "policy_seed_base": POLICY_SEED_BASE, **state.summary()}
    write_json_atomic(state_path(run_dir, "D_H"), final)
    print(
        f"Merged {len(ordered)} D_H attempts from {len(pool)} seed shards -> "
        f"{merged_path} (status={final['status']}, "
        f"M_H={final['successful_trajectories']}, U={final['unique_success_seeds']})"
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

    em_ok = all(statuses[g] == "complete" for g in ("D_E", "D_M"))
    em_n_eff_ok = all(groups[g]["n_eff_gate_pass"] for g in ("D_E", "D_M"))
    if statuses["D_H"] not in ("budget_exhausted", "collecting"):
        raise RuntimeError(
            f"D_H state has status {statuses['D_H']!r}; under protocol.t1.v1.2 "
            "D_H must run in fixed_budget mode (budget_exhausted when done). "
            "A v1.1-era state file cannot be judged by the v1.2 gate."
        )
    dh_done = statuses["D_H"] == "budget_exhausted"

    m = {g: groups[g]["successful_trajectories"] for g in GROUPS}
    joint_gate = evaluate_joint_gate(
        rules["joint_gate_G_H"], m, groups["D_H"]["unique_success_seeds"],
        groups["D_H"]["n_eff"],
    )

    if not (em_ok and em_n_eff_ok and dh_done):
        verdict = (
            "FAIL — E/M incomplete or D_H budget not exhausted; see group "
            "statuses. Per the frozen rules T1d halts for this task."
        )
    elif joint_gate["G_H"]:
        verdict = (
            "PASS — D_E/D_M complete, D_H joint gate G_H = 1 (source-"
            "admissible); T1d may proceed once D_Q is collected"
        )
    else:
        verdict = (
            "FAIL — D_H joint gate G_H = 0: source-infeasibility certificate; "
            "T1d halts for this task (no post-hoc seeds/budget/dose changes)"
        )

    report = {
        "record": f"capability_transport.t1c_collection.{args.task_name}.v1_2",
        "protocol_revision": rules["protocol_revision"],
        "task": args.task_name,
        "provenance": provenance,
        "groups": {
            group: {
                key: payload[key]
                for key in (
                    "mode",
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
        "joint_gate_G_H": joint_gate,
        "verdict": verdict,
    }
    if verdict.startswith("FAIL") and dh_done and not joint_gate["G_H"]:
        failed = [
            name for name, part in joint_gate["parts"].items() if not part["pass"]
        ]
        report["infeasibility_certificate"] = {
            "task": args.task_name,
            "feasible": False,
            "binding_constraints": failed,
            "missing_source": "D_H_supported",
            "q_star_lower_bound": "not_computable — requires the T1d dual "
            "variables; this certificate records the acquisition-stage gate "
            "failure per protocol.t1d.v1.1 gate_3 schema",
            "evidence": {
                "group_sizes": m,
                "U_H": groups["D_H"]["unique_success_seeds"],
                "n_eff_H": groups["D_H"]["n_eff"],
                "q": joint_gate["q"],
            },
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
        description="T1c round-robin success-first source collection (protocol.t1.v1.2)."
    )
    parser.add_argument("--task", dest="task_name", required=True)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("--ckpt-path", type=Path)
    parser.add_argument("--normalizer-zarr", type=Path)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=repo_path("experiments", "capability_transport", "protocol.t1.v1.2.json"),
    )
    parser.add_argument("--groups-file", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--only-seeds",
        nargs="+",
        type=int,
        help="D_H only: collect a single frozen supported-hard seed in this "
        "process (seed-parallel packing). Not legal for D_E/D_M.",
    )
    parser.add_argument(
        "--merge-dh-shards",
        action="store_true",
        help="Merge per-seed D_H manifests into manifest_D_H.jsonl / "
        "state_D_H.json after all seed workers finish.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Posterior-MEAN smoke test (logic/resume only; NOT predictive — "
        "use --mc-feasibility for forecasts). No GPU.",
    )
    parser.add_argument(
        "--mc-feasibility",
        action="store_true",
        help="Posterior-predictive Monte Carlo of the D_H joint gate G_H "
        "(p_s ~ Beta(1+s, 1+R-s) per seed); writes "
        "t1c_dh_mc_feasibility.<task>.v1.json. No GPU, no run dir needed.",
    )
    parser.add_argument("--mc-replicates", type=int)
    parser.add_argument("--mc-output", type=Path)
    parser.add_argument(
        "--report",
        action="store_true",
        help="Merge the three group states into t1c_collection_report.json, "
        "evaluate the joint gate G_H, and append the efficiency ledger entry.",
    )
    parser.add_argument(
        "--run-attempt",
        action="store_true",
        help="Run a single (env_seed, attempt_index) for round-parallel D_E/D_M.",
    )
    parser.add_argument("--env-seed", type=int)
    parser.add_argument("--attempt-index", type=int)
    parser.add_argument("--round-index", type=int)
    parser.add_argument(
        "--parallel-rounds",
        action="store_true",
        help="D_E/D_M: parallelize each round-robin wave across GPUs, commit "
        "in frozen seed order, stop at N_target (serial earliest-stop equivalent).",
    )
    parser.add_argument(
        "--gpu-ids",
        default=os.environ.get("T1_GPU_IDS", "0 1 2 3 4 5 6 7"),
        help="Physical GPU ids for --parallel-rounds (space-separated).",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=int(os.environ.get("T1_WORKERS_PER_GPU", "3")),
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

    provenance = {
        "groups_file": str(groups_file),
        "groups_file_sha256": groups_sha,
        "protocol_file": str(args.protocol),
        "protocol_sha256": protocol_sha,
        "ckpt_path": str(args.ckpt_path) if args.ckpt_path else None,
        "ckpt_sha256": (
            sha256_file(args.ckpt_path)
            if args.ckpt_path and args.ckpt_path.is_file()
            else None
        ),
        "policy_seed_base": POLICY_SEED_BASE,
    }

    if args.mc_feasibility:
        mc_feasibility(args, rules, groups_payload, provenance)
        return

    if not args.run_dir:
        parser.error("--run-dir is required unless --mc-feasibility is set")
    if not args.ckpt_path:
        parser.error("--ckpt-path is required unless --mc-feasibility is set")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.run_dir / "provenance.json", provenance)

    if args.merge_dh_shards:
        merge_dh_shards(args, rules, groups_payload, args.run_dir)
        return
    if args.report:
        write_report(args, rules, args.run_dir, provenance)
        return
    if args.parallel_rounds:
        if not args.group:
            parser.error("--parallel-rounds requires --group")
        parallel_target_rounds(args, rules, groups_payload, args.run_dir)
        return
    if args.run_attempt:
        if args.group is None or args.env_seed is None or args.attempt_index is None or args.round_index is None:
            parser.error("--run-attempt requires --group --env-seed --attempt-index --round-index")
        run_one_attempt(args, rules, groups_payload, args.run_dir)
        return
    if not args.group:
        parser.error("--group is required unless --report, --merge-dh-shards, --parallel-rounds, or --run-attempt is set")
    if not args.dry_run and not args.ckpt_path.is_file():
        raise RuntimeError(f"missing checkpoint: {args.ckpt_path}")
    collect_group(args, rules, groups_payload, args.run_dir)


if __name__ == "__main__":
    main()
