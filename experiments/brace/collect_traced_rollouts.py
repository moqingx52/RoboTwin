#!/usr/bin/env python3
"""Collect BRACE traced rollouts with schema v2 extensions in HDF5."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

PHASE1_DIR = Path(__file__).resolve().parents[1] / "phase1"
REPO_ROOT = Path(__file__).resolve().parents[2]
for import_path in (REPO_ROOT, PHASE1_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from common import (  # noqa: E402
    add_common_args,
    append_jsonl,
    load_task_args,
    make_task_env,
    read_json,
    repo_path,
    split_range,
    write_json,
)


def path_for_manifest(path: Path) -> str:
    abs_path = path.resolve()
    repo_root = repo_path().resolve()
    try:
        return str(abs_path.relative_to(repo_root))
    except ValueError:
        return str(abs_path)


def _read_n_action_steps(action_dim: int) -> int:
    import yaml

    config_path = repo_path("policy", "DP", "diffusion_policy", "config", f"robot_dp_{action_dim}.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return int(payload["n_action_steps"])


def make_model_args(task_name, task_config, ckpt_setting, expert_data_num, train_seed, checkpoint_num, action_dim):
    arm_dim = (action_dim - 2) // 2
    return {
        "task_name": task_name,
        "task_config": task_config,
        "ckpt_setting": ckpt_setting,
        "expert_data_num": expert_data_num,
        "seed": train_seed,
        "checkpoint_num": checkpoint_num,
        "left_arm_dim": arm_dim,
        "right_arm_dim": arm_dim,
    }


def rollout_once(
    env,
    model,
    env_args,
    env_seed,
    rollout_id,
    episode_idx,
    save_root,
    save_failures=False,
    snapshots_per_trajectory=3,
    action_dim=14,
):
    from policy.DP.deploy_policy import encode_obs

    episode_name = f"episode_{env_seed}_{rollout_id}"
    tmp_dir = save_root / ".tmp" / episode_name
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    run_args = dict(env_args)
    run_args.update(
        {
            "need_plan": False,
            "save_data": True,
            "save_path": str(tmp_dir),
            "eval_mode": True,
            "render_freq": 0,
            "n_action_steps": _read_n_action_steps(action_dim),
        }
    )
    if run_args.get("save_freq") is None:
        run_args["save_freq"] = 15

    try:
        if hasattr(model, "set_generator"):
            gen = torch.Generator(device="cuda:0")
            gen.manual_seed(int(rollout_id))
            model.set_generator(gen)

        model.reset_obs()
        env.setup_demo(now_ep_num=episode_idx, seed=env_seed, is_test=True, **run_args)
        env.reset_brace_trace_buffers()
        env.set_instruction("brace traced rollout")

        success = False
        chunk_index = 0
        while env.take_action_cnt < env.step_lim:
            observation = env.get_obs()
            obs = encode_obs(observation)
            actions = model.get_action(obs)
            env.record_policy_chunk(actions[0], chunk_index)
            for action in actions:
                env.take_action(action)
                observation = env.get_obs()
                obs = encode_obs(observation)
                model.update_obs(obs)
                if env.eval_success:
                    success = True
                    break
            chunk_index += 1
            if success:
                break

        raw_episode = tmp_dir / "data" / f"episode{episode_idx}.hdf5"
        success_path = None
        failure_path = None
        final_path = None
        if success:
            env.merge_pkl_to_hdf5_video()
            success_dir = save_root / "successes"
            success_dir.mkdir(parents=True, exist_ok=True)
            success_path = success_dir / f"{episode_name}.hdf5"
            shutil.move(str(raw_episode), str(success_path))
            final_path = success_path
        elif save_failures:
            env.merge_pkl_to_hdf5_video()
            if raw_episode.is_file():
                failure_dir = save_root / "failures"
                failure_dir.mkdir(parents=True, exist_ok=True)
                failure_path = failure_dir / f"{episode_name}.hdf5"
                shutil.move(str(raw_episode), str(failure_path))
                final_path = failure_path

        if final_path is not None:
            env.finalize_brace_trace_to_hdf5(final_path, snapshots_per_trajectory=snapshots_per_trajectory)

        return {
            "env_seed": int(env_seed),
            "rollout_id": int(rollout_id),
            "policy_seed": int(rollout_id),
            "episode_idx": int(episode_idx),
            "success": bool(success),
            "hdf5_path": path_for_manifest(success_path) if success_path else None,
            "failure_hdf5_path": path_for_manifest(failure_path) if failure_path else None,
            "steps": int(env.take_action_cnt),
            "physics_steps": int(env.physics_step),
            "trace_schema_version": 2 if final_path is not None else None,
        }
    finally:
        try:
            env.close_env()
        except Exception:
            pass
        try:
            env.remove_data_cache()
        except Exception:
            pass
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def iter_manifest(path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_manifest_atomic(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def manifest_path_exists(path_value):
    if not path_value:
        return False
    path = Path(path_value)
    if not path.is_absolute():
        path = repo_path(path_value)
    return path.is_file()


def summarize_manifest(manifest_paths, seeds, rollouts_per_seed):
    rows = []
    for manifest_path in manifest_paths:
        rows.extend(iter_manifest(manifest_path))
    by_seed = {}
    for seed in seeds:
        seed_rows = [row for row in rows if row["env_seed"] == seed]
        success_count = sum(1 for row in seed_rows if row.get("success"))
        by_seed[str(seed)] = {
            "attempts": len(seed_rows),
            "successes": success_count,
            "j_hat": success_count / rollouts_per_seed if rollouts_per_seed else 0.0,
        }
    return by_seed


def main():
    parser = argparse.ArgumentParser(description="Collect BRACE traced rollouts (schema v2 HDF5).")
    add_common_args(parser)
    parser.set_defaults(task_config="demo_brace_trace")
    parser.add_argument("--seeds-file", type=Path)
    parser.add_argument("--rollouts-per-seed", type=int, default=8)
    parser.add_argument("--ckpt-setting", default="demo_clean")
    parser.add_argument("--expert-data-num", type=int, default=200)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--checkpoint-num", type=int, default=600)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--output-dir", type=Path, default=repo_path("experiments", "brace", "rollouts_traced"))
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-failures", action="store_true", default=True)
    parser.add_argument("--snapshots-per-trajectory", type=int, default=3)
    parser.add_argument("--max-trajectories", type=int, default=None, help="Optional cap for smoke collection.")
    parser.add_argument("--env-seeds", nargs="*", type=int, default=None, help="Optional explicit env seed allowlist.")
    parser.add_argument("--rollout-ids", nargs="*", type=int, default=None, help="Optional rollout id allowlist.")
    args = parser.parse_args()

    seeds_file = args.seeds_file or repo_path("experiments", "phase1", "seeds", f"{args.task_name}_seeds.json")
    seeds_payload = read_json(seeds_file)
    seeds = split_range(seeds_payload["train_rollout"], args.shard_id, args.num_shards)
    if args.env_seeds is not None:
        allow = {int(seed) for seed in args.env_seeds}
        seeds = [seed for seed in seeds if seed in allow]

    save_root = (args.output_dir / args.task_name).resolve()
    if args.num_shards > 1:
        manifest_path = save_root / f"manifest_shard_{args.shard_id:02d}_of_{args.num_shards:02d}.jsonl"
        stats_path = save_root / f"seed_stats_shard_{args.shard_id:02d}_of_{args.num_shards:02d}.json"
    else:
        manifest_path = save_root / "manifest.jsonl"
        stats_path = save_root / "seed_stats.json"
    save_root.mkdir(parents=True, exist_ok=True)

    existing_rows = iter_manifest(manifest_path)
    done = set()
    failure_backfill_indices = {}
    if args.resume and manifest_path.exists():
        for idx, row in enumerate(existing_rows):
            key = (row["env_seed"], row["rollout_id"])
            needs_failure_backfill = (
                args.save_failures
                and not row.get("success", False)
                and not manifest_path_exists(row.get("failure_hdf5_path"))
            )
            needs_trace_backfill = row.get("trace_schema_version") != 2
            if needs_failure_backfill or needs_trace_backfill:
                failure_backfill_indices[key] = idx
            else:
                done.add(key)

    work_items = []
    rollout_ids = (
        [int(value) for value in args.rollout_ids]
        if args.rollout_ids is not None
        else list(range(args.rollouts_per_seed))
    )
    for env_seed in seeds:
        for rollout_id in rollout_ids:
            key = (env_seed, rollout_id)
            if key in done:
                continue
            work_items.append(key)
    if args.max_trajectories is not None:
        work_items = work_items[: args.max_trajectories]

    os.chdir(repo_path())
    env_args = load_task_args(args.task_name, args.task_config)
    from policy.DP.deploy_policy import get_model

    model_args = make_model_args(
        args.task_name,
        args.task_config,
        args.ckpt_setting,
        args.expert_data_num,
        args.train_seed,
        args.checkpoint_num,
        args.action_dim,
    )
    model = get_model(model_args)

    episode_idx = len(existing_rows)
    for env_seed, rollout_id in work_items:
        env = make_task_env(args.task_name)
        key = (env_seed, rollout_id)
        backfill_idx = failure_backfill_indices.get(key)
        run_episode_idx = (
            int(existing_rows[backfill_idx].get("episode_idx", episode_idx))
            if backfill_idx is not None
            else episode_idx
        )
        row = rollout_once(
            env,
            model,
            env_args,
            env_seed,
            rollout_id,
            run_episode_idx,
            save_root,
            args.save_failures,
            snapshots_per_trajectory=args.snapshots_per_trajectory,
            action_dim=args.action_dim,
        )
        if backfill_idx is not None:
            existing_rows[backfill_idx] = row
            write_manifest_atomic(manifest_path, existing_rows)
        else:
            append_jsonl(manifest_path, row)
            existing_rows.append(row)
            episode_idx += 1
        print(
            f"[{args.task_name}] seed={env_seed} rollout={rollout_id} "
            f"success={row['success']} physics_steps={row.get('physics_steps')} "
            f"backfill={backfill_idx is not None}"
        )

    write_json(stats_path, summarize_manifest([manifest_path], seeds, args.rollouts_per_seed))
    print(f"Wrote {manifest_path}")
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
