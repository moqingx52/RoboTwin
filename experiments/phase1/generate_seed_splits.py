#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

from common import add_common_args, load_task_args, make_task_env, repo_path, write_json


def is_feasible(task_name, task_config, seed):
    from envs.utils.create_actor import UnStableError

    args = load_task_args(task_name, task_config)
    args["need_plan"] = True
    args["save_data"] = False
    args["collect_data"] = False
    args["save_path"] = str(repo_path("experiments", "phase1", ".tmp_seed_check", task_name))
    env = make_task_env(task_name)
    try:
        env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
        env.play_once()
        return bool(env.plan_success and env.check_success())
    except UnStableError:
        return False
    except Exception as exc:
        print(f"[WARN] seed={seed} failed feasibility check: {exc}")
        return False
    finally:
        try:
            env.close_env()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Generate fixed feasible seed splits for phase1 DP experiments.")
    add_common_args(parser)
    parser.add_argument("--start-seed", type=int, default=100000)
    parser.add_argument("--num-train", type=int, default=100)
    parser.add_argument("--num-eval", type=int, default=100)
    parser.add_argument("--num-reserve", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=20000)
    parser.add_argument("--output-dir", type=Path, default=repo_path("experiments", "phase1", "seeds"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    need = args.num_train + args.num_eval + args.num_reserve
    feasible = []
    candidate = args.start_seed
    attempts = 0

    if args.dry_run:
        feasible = list(range(args.start_seed, args.start_seed + need))
    else:
        while len(feasible) < need and attempts < args.max_candidates:
            ok = is_feasible(args.task_name, args.task_config, candidate)
            attempts += 1
            if ok:
                feasible.append(candidate)
                print(f"[OK] {args.task_name} feasible seed {candidate} ({len(feasible)}/{need})")
            else:
                print(f"[SKIP] {args.task_name} infeasible seed {candidate}")
            candidate += 1
            time.sleep(0.05)

    if len(feasible) < need:
        raise RuntimeError(f"Only found {len(feasible)} feasible seeds, need {need}.")

    payload = {
        "train_rollout": feasible[:args.num_train],
        "eval_id": feasible[args.num_train:args.num_train + args.num_eval],
        "reserve": feasible[args.num_train + args.num_eval:],
        "meta": {
            "task_name": args.task_name,
            "task_config": args.task_config,
            "st_seed_start": args.start_seed,
            "num_train": args.num_train,
            "num_eval": args.num_eval,
            "num_reserve": args.num_reserve,
            "dry_run": args.dry_run,
        },
    }
    out_path = args.output_dir / f"{args.task_name}_seeds.json"
    write_json(out_path, payload)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

