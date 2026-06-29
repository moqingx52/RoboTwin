#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

from common import add_common_args, load_task_args, make_task_env, read_json, repo_path, write_json

sys.path.append(str(repo_path()))


def load_dp_model(ckpt_path, action_dim):
    from policy.DP.dp_model import DP

    config_path = repo_path("policy", "DP", "diffusion_policy", "config", f"robot_dp_{action_dim}.yaml")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return DP(str(ckpt_path), n_obs_steps=cfg["n_obs_steps"], n_action_steps=cfg["n_action_steps"])


def evaluate_once(task_name, env_args, model, env_seed, policy_seed):
    from policy.DP.deploy_policy import encode_obs

    env = make_task_env(task_name)
    try:
        if hasattr(model, "set_generator"):
            gen = torch.Generator(device="cuda:0")
            gen.manual_seed(int(policy_seed))
            model.set_generator(gen)

        model.reset_obs()
        env.setup_demo(now_ep_num=0, seed=env_seed, is_test=True, **env_args)
        env.set_instruction("phase1 dp eval")
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
        return {"success": bool(success), "steps": int(env.take_action_cnt)}
    finally:
        try:
            env.close_env()
        except Exception:
            pass


def summarize(rows, split_name):
    split_rows = [row for row in rows if row["split"] == split_name]
    if not split_rows:
        return {}
    by_seed = {}
    for row in split_rows:
        by_seed.setdefault(row["env_seed"], []).append(row["success"])
    solved = sum(1 for vals in by_seed.values() if any(vals))
    return {
        "episodes": len(split_rows),
        "seeds": len(by_seed),
        "mean_sr": sum(row["success"] for row in split_rows) / len(split_rows),
        "solved_coverage": solved / len(by_seed),
    }


def hard_seeds_from_stats(seed_stats, count=20):
    items = sorted(seed_stats.items(), key=lambda kv: (kv[1]["j_hat"], int(kv[0])))
    return [int(seed) for seed, _ in items[:count]]


def main():
    parser = argparse.ArgumentParser(description="Evaluate DP checkpoints per seed for phase1 diagnostics.")
    add_common_args(parser)
    parser.add_argument(
        "--variant",
        required=True,
        choices=("base", "expert_only", "success", "seed_balanced", "difficulty_weighted"),
    )
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--seeds-file", type=Path)
    parser.add_argument(
        "--hard-seeds-file",
        type=Path,
        help="Optional JSON list (or object with hard_seeds) defining a shared held-out hard split.",
    )
    parser.add_argument("--rollout-dir", type=Path, default=repo_path("experiments", "phase1", "rollouts"))
    parser.add_argument("--output-dir", type=Path, default=repo_path("experiments", "phase1", "eval_results"))
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--id-repeats", type=int, default=3)
    parser.add_argument("--train-repeats", type=int, default=3)
    parser.add_argument("--hard-repeats", type=int, default=8)
    parser.add_argument(
        "--policy-seed-offset",
        type=int,
        default=0,
        help="Offset stochastic policy seeds; use a fresh offset after selecting a hard split.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds_file = args.seeds_file or repo_path("experiments", "phase1", "seeds", f"{args.task_name}_seeds.json")
    seed_payload = read_json(seeds_file)
    seed_stats_path = args.rollout_dir / args.task_name / "seed_stats.json"
    seed_stats = read_json(seed_stats_path) if seed_stats_path.exists() else {}
    if args.hard_seeds_file:
        hard_payload = read_json(args.hard_seeds_file)
        hard_seeds = hard_payload["hard_seeds"] if isinstance(hard_payload, dict) else hard_payload
        hard_seeds = [int(seed) for seed in hard_seeds]
        hard_seed_source = str(args.hard_seeds_file)
    else:
        hard_seeds = hard_seeds_from_stats(seed_stats) if seed_stats else seed_payload["train_rollout"][:20]
        hard_seed_source = "train_rollout_seed_stats"

    rows = []
    if args.dry_run:
        for split, seeds, repeats in (
            ("id_heldout", seed_payload["eval_id"], args.id_repeats),
            ("train_seen", seed_payload["train_rollout"], args.train_repeats),
            ("hard_20", hard_seeds, args.hard_repeats),
        ):
            for env_seed in seeds:
                for repeat in range(repeats):
                    rows.append(
                        {
                            "split": split,
                            "env_seed": int(env_seed),
                            "repeat": repeat,
                            "policy_seed": args.policy_seed_offset + repeat,
                            "success": repeat % 2 == 0,
                            "steps": 0,
                        }
                    )
    else:
        os.chdir(repo_path())
        env_args = load_task_args(args.task_name, args.task_config)
        env_args["need_plan"] = False
        env_args["save_data"] = False
        env_args["render_freq"] = 0
        model = load_dp_model(args.ckpt_path, args.action_dim)
        for split, seeds, repeats in (
            ("id_heldout", seed_payload["eval_id"], args.id_repeats),
            ("train_seen", seed_payload["train_rollout"], args.train_repeats),
            ("hard_20", hard_seeds, args.hard_repeats),
        ):
            for env_seed in seeds:
                for repeat in range(repeats):
                    policy_seed = args.policy_seed_offset + repeat
                    result = evaluate_once(args.task_name, env_args, model, env_seed, policy_seed)
                    row = {
                        "split": split,
                        "env_seed": int(env_seed),
                        "repeat": repeat,
                        "policy_seed": policy_seed,
                        **result,
                    }
                    rows.append(row)
                    print(f"[{args.task_name}/{args.variant}] {split} seed={env_seed} repeat={repeat} success={row['success']}")

    summary = {
        "task_name": args.task_name,
        "task_config": args.task_config,
        "variant": args.variant,
        "ckpt_path": str(args.ckpt_path),
        "splits": {
            "id_heldout": summarize(rows, "id_heldout"),
            "train_seen": summarize(rows, "train_seen"),
            "hard_20": summarize(rows, "hard_20"),
        },
        "hard_seeds": hard_seeds,
        "hard_seed_source": hard_seed_source,
        "rows": rows,
    }
    out_path = args.output_dir / args.task_name / f"{args.variant}.json"
    write_json(out_path, summary)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
