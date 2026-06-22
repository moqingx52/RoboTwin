#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
from pathlib import Path

import torch

from common import (
    add_common_args,
    append_jsonl,
    load_task_args,
    make_task_env,
    read_json,
    repo_path,
    split_range,
    write_json,
)

sys.path.append(str(repo_path()))


def make_model_args(task_name, task_config, ckpt_setting, expert_data_num, train_seed, checkpoint_num, action_dim):
    # Aloha-AgileX DP uses 6 arm joints + 1 gripper per arm.
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


def rollout_once(env, model, env_args, env_seed, rollout_id, episode_idx, save_root):
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
        env.set_instruction("phase1 dp rollout")

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

        raw_episode = tmp_dir / "data" / f"episode{episode_idx}.hdf5"
        success_path = None
        if success:
            env.merge_pkl_to_hdf5_video()
            success_dir = save_root / "successes"
            success_dir.mkdir(parents=True, exist_ok=True)
            success_path = success_dir / f"{episode_name}.hdf5"
            shutil.move(str(raw_episode), str(success_path))

        return {
            "env_seed": int(env_seed),
            "rollout_id": int(rollout_id),
            "policy_seed": int(rollout_id),
            "episode_idx": int(episode_idx),
            "success": bool(success),
            "hdf5_path": str(success_path.relative_to(repo_path())) if success_path else None,
            "steps": int(env.take_action_cnt),
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


def summarize_manifest(manifest_paths, seeds, rollouts_per_seed):
    rows = []
    for manifest_path in manifest_paths:
        rows.extend(iter_manifest(manifest_path))
    by_seed = {}
    for seed in seeds:
        seed_rows = [row for row in rows if row["env_seed"] == seed]
        success_count = sum(1 for row in seed_rows if row["success"])
        by_seed[str(seed)] = {
            "attempts": len(seed_rows),
            "successes": success_count,
            "j_hat": success_count / rollouts_per_seed if rollouts_per_seed else 0.0,
        }
    return by_seed


def iter_manifest(path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                import json
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Collect DP policy rollouts for phase1 success-filtered SFT.")
    add_common_args(parser)
    parser.add_argument("--seeds-file", type=Path)
    parser.add_argument("--rollouts-per-seed", type=int, default=8)
    parser.add_argument("--ckpt-setting", default="demo_clean")
    parser.add_argument("--expert-data-num", type=int, default=50)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--checkpoint-num", type=int, default=600)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--output-dir", type=Path, default=repo_path("experiments", "phase1", "rollouts"))
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds_file = args.seeds_file or repo_path("experiments", "phase1", "seeds", f"{args.task_name}_seeds.json")
    seeds_payload = read_json(seeds_file)
    seeds = split_range(seeds_payload["train_rollout"], args.shard_id, args.num_shards)

    save_root = args.output_dir / args.task_name
    if args.num_shards > 1:
        manifest_path = save_root / f"manifest_shard_{args.shard_id:02d}_of_{args.num_shards:02d}.jsonl"
        stats_path = save_root / f"seed_stats_shard_{args.shard_id:02d}_of_{args.num_shards:02d}.json"
    else:
        manifest_path = save_root / "manifest.jsonl"
        stats_path = save_root / "seed_stats.json"
    save_root.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.resume and manifest_path.exists():
        for row in iter_manifest(manifest_path):
            done.add((row["env_seed"], row["rollout_id"]))

    if args.dry_run:
        for env_seed in seeds:
            for rollout_id in range(args.rollouts_per_seed):
                row = {
                    "env_seed": int(env_seed),
                    "rollout_id": int(rollout_id),
                    "policy_seed": int(rollout_id),
                    "episode_idx": len(done),
                    "success": rollout_id % 2 == 0,
                    "hdf5_path": None,
                    "steps": 0,
                    "dry_run": True,
                }
                append_jsonl(manifest_path, row)
        write_json(stats_path, summarize_manifest([manifest_path], seeds, args.rollouts_per_seed))
        print(f"Dry-run wrote {manifest_path}")
        return

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

    episode_idx = len(iter_manifest(manifest_path))
    for env_seed in seeds:
        for rollout_id in range(args.rollouts_per_seed):
            key = (env_seed, rollout_id)
            if key in done:
                continue
            env = make_task_env(args.task_name)
            row = rollout_once(env, model, env_args, env_seed, rollout_id, episode_idx, save_root)
            append_jsonl(manifest_path, row)
            print(f"[{args.task_name}] seed={env_seed} rollout={rollout_id} success={row['success']}")
            episode_idx += 1

    write_json(stats_path, summarize_manifest([manifest_path], seeds, args.rollouts_per_seed))
    print(f"Wrote {manifest_path}")
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()

