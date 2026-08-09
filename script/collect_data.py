import sys

sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os
import time
import hashlib
from argparse import ArgumentParser
from pathlib import Path

from experiments.brace.premotion_retry import (
    append_jsonl,
    materialize_saved_seed_trajectory,
    resolve_max_attempts,
    utc_now,
)

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def reject_success_first_materialization(args, seed_list, episode_idx, seed, error):
    """Remove one replay-failed accepted seed and compact pending trajectories."""
    save_path = Path(args["save_path"])
    append_jsonl(
        save_path / "expert_acquisition_attempts.jsonl",
        {
            "schema_version": 2,
            "stage": "base_expert_hdf5_materialization",
            "created_at": utc_now(),
            "task": args["task_name"],
            "task_config": args["task_config"],
            "seed": int(seed),
            "candidate_seed_start": int(args.get("candidate_seed_start", 0)),
            "candidate_seed_stop": (
                int(args.get("candidate_seed_start", 0)) + int(args["candidate_seed_limit"])
                if args.get("candidate_seed_limit") is not None else None
            ),
            "target_successes": int(args["episode_num"]),
            "accepted_episode_idx": int(episode_idx),
            "passed": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "git_commit": os.environ.get("BRACE_GIT_COMMIT"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pid": os.getpid(),
        },
    )
    for path in (
        save_path / "data" / f"episode{episode_idx}.hdf5",
        save_path / "_traj_data" / f"episode{episode_idx}.pkl",
    ):
        if path.exists():
            path.unlink()

    old_count = len(seed_list)
    seed_list.pop(episode_idx)
    trajectory_dir = save_path / "_traj_data"
    for old_idx in range(episode_idx + 1, old_count):
        old_path = trajectory_dir / f"episode{old_idx}.pkl"
        if old_path.exists():
            old_path.replace(trajectory_dir / f"episode{old_idx - 1}.pkl")
    (save_path / "seed.txt").write_text(
        "".join(f"{value} " for value in seed_list), encoding="utf-8"
    )

    info_path = save_path / "scene_info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info.pop(f"episode_{episode_idx}", None)
        shifted = {}
        for key, value in info.items():
            if key.startswith("episode_") and key[8:].isdigit() and int(key[8:]) > episode_idx:
                key = f"episode_{int(key[8:]) - 1}"
            shifted[key] = value
        info_path.write_text(json.dumps(shifted, ensure_ascii=False, indent=4), encoding="utf-8")


def main(task_name=None, task_config=None):

    task = class_decorator(task_name)
    config_path = f"./task_config/{task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    # BRACE multitask v2 uses RoboTwin's native success-first collection, but
    # makes its target and bounded candidate stream explicit and resumable.
    # These overrides are intentionally opt-in so ordinary RoboTwin collection
    # and the frozen multitask-v1 saved-seed path are unchanged.
    if os.environ.get("BRACE_SUCCESS_FIRST", "0") == "1":
        args["episode_num"] = int(os.environ.get("BRACE_EXPERT_TARGET_SUCCESSES", "200"))
        args["use_seed"] = False
        args["success_first"] = True
        args["candidate_seed_start"] = int(os.environ["BRACE_CANDIDATE_SEED_START"])
        args["candidate_seed_limit"] = int(os.environ.get("BRACE_CANDIDATE_SEED_LIMIT", "5000"))
        if args["episode_num"] < 1 or args["candidate_seed_limit"] < args["episode_num"]:
            raise RuntimeError("invalid BRACE success-first target/candidate limit")

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
    run(task, args)


def run(TASK_ENV, args):
    epid, suc_num, fail_num, seed_list = 0, 0, 0, []

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")

    # =========== Collect Seed ===========
    os.makedirs(args["save_path"], exist_ok=True)

    if not args["use_seed"]:
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
        args["need_plan"] = True

        attempt_log = Path(args["save_path"]) / "expert_acquisition_attempts.jsonl"
        candidate_start = int(args.get("candidate_seed_start", 0))
        candidate_limit = args.get("candidate_seed_limit")
        candidate_stop = candidate_start + int(candidate_limit) if candidate_limit is not None else None
        epid = candidate_start

        if os.path.exists(os.path.join(args["save_path"], "seed.txt")):
            with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
                seed_list = file.read().split()
                if len(seed_list) != 0:
                    seed_list = [int(i) for i in seed_list]
                    suc_num = len(seed_list)
                    epid = max(epid, max(seed_list) + 1)
            print(f"Exist seed file, Start from: {epid} / {suc_num}")

        # A failed candidate is absent from seed.txt. Recover the last attempted
        # seed from the append-only log so resume never silently retries or
        # changes the preregistered candidate order.
        if attempt_log.is_file():
            for line in attempt_log.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    epid = max(epid, int(json.loads(line)["seed"]) + 1)

        while suc_num < args["episode_num"]:
            if candidate_stop is not None and epid >= candidate_stop:
                raise RuntimeError(
                    f"expert_acquisition_infeasible: collected {suc_num}/{args['episode_num']} "
                    f"successes after candidate seeds [{candidate_start},{candidate_stop})"
                )
            attempt_row = {
                "schema_version": 2,
                "stage": "base_expert_success_first",
                "created_at": utc_now(),
                "task": args["task_name"],
                "task_config": args["task_config"],
                "seed": epid,
                "candidate_seed_start": candidate_start,
                "candidate_seed_stop": candidate_stop,
                "target_successes": int(args["episode_num"]),
                "accepted_episode_idx": None,
                "plan_success": False,
                "check_success": False,
                "trajectory_saved": False,
                "passed": False,
                "error_type": None,
                "error_message": None,
                "git_commit": os.environ.get("BRACE_GIT_COMMIT"),
                "task_config_sha256": hashlib.sha256(
                    Path(f"task_config/{args['task_config']}.yml").read_bytes()
                ).hexdigest(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "pid": os.getpid(),
            }
            try:
                TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)
                TASK_ENV.play_once()
                attempt_row["plan_success"] = bool(TASK_ENV.plan_success)
                if attempt_row["plan_success"]:
                    attempt_row["check_success"] = bool(TASK_ENV.check_success())

                if attempt_row["plan_success"] and attempt_row["check_success"]:
                    print(f"simulate data episode {suc_num} success! (seed = {epid})")
                    seed_list.append(epid)
                    TASK_ENV.save_traj_data(suc_num)
                    attempt_row["accepted_episode_idx"] = suc_num
                    attempt_row["trajectory_saved"] = True
                    attempt_row["passed"] = True
                    suc_num += 1
                else:
                    print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                    attempt_row["error_type"] = (
                        "expert_plan_failed" if not attempt_row["plan_success"] else "expert_check_failed"
                    )
                    fail_num += 1

                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                attempt_row["error_type"] = type(e).__name__
                attempt_row["error_message"] = str(e)
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(0.3)
            except Exception as e:
                # stack_trace = traceback.format_exc()
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                attempt_row["error_type"] = type(e).__name__
                attempt_row["error_message"] = str(e)
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(1)

            if args.get("success_first"):
                append_jsonl(attempt_log, attempt_row)
            epid += 1

            with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
                for sed in seed_list:
                    file.write("%s " % sed)

        print(f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]

        if len(seed_list) < args["episode_num"]:
            raise RuntimeError(
                f"Saved seed list has {len(seed_list)} entries, "
                f"but episode_num={args['episode_num']}"
            )

        # A saved seed list fixes which scenes to use; it does not imply that
        # the corresponding pre-motion trajectories already exist.  Generate
        # only missing trajectories so fixed-seed collection can start from a
        # fresh data directory and can also resume safely.
        print("\033[93m" + "[Prepare Missing Pre Motion Data]" + "\033[0m")
        args["need_plan"] = True
        premotion_max_attempts, retry_amendment_path = resolve_max_attempts(
            cwd=Path.cwd(), task_name=args["task_name"]
        )
        print(
            f"Pre-motion max attempts per fixed seed: {premotion_max_attempts} "
            f"(amendment={retry_amendment_path})"
        )
        for episode_idx, seed in enumerate(seed_list[: args["episode_num"]]):
            traj_path = os.path.join(
                args["save_path"], "_traj_data", f"episode{episode_idx}.pkl"
            )
            data_path = os.path.join(
                args["save_path"], "data", f"episode{episode_idx}.hdf5"
            )
            if os.path.exists(traj_path) or os.path.exists(data_path):
                continue

            result = materialize_saved_seed_trajectory(
                TASK_ENV,
                args,
                seed=seed,
                episode_idx=episode_idx,
                max_attempts=premotion_max_attempts,
                amendment_path=retry_amendment_path,
            )
            print(
                f"prepared pre-motion episode {episode_idx} (seed = {seed}, "
                f"attempt = {result['attempt']}/{premotion_max_attempts})"
            )

    # =========== Collect Data ===========

    if args["collect_data"]:
        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        args["need_plan"] = False
        args["render_freq"] = 0
        args["save_data"] = True

        clear_cache_freq = args["clear_cache_freq"]

        st_idx = 0

        def exist_hdf5(idx):
            file_path = os.path.join(args["save_path"], 'data', f'episode{idx}.hdf5')
            return os.path.exists(file_path)

        while exist_hdf5(st_idx):
            st_idx += 1

        for episode_idx in range(st_idx, args["episode_num"]):
            print(f"\033[34mTask name: {args['task_name']}\033[0m")
            try:
                TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed_list[episode_idx], **args)

                traj_data = TASK_ENV.load_tran_data(episode_idx)
                args["left_joint_path"] = traj_data["left_joint_path"]
                args["right_joint_path"] = traj_data["right_joint_path"]
                TASK_ENV.set_path_lst(args)

                info_file_path = os.path.join(args["save_path"], "scene_info.json")

                if not os.path.exists(info_file_path):
                    with open(info_file_path, "w", encoding="utf-8") as file:
                        json.dump({}, file, ensure_ascii=False)

                with open(info_file_path, "r", encoding="utf-8") as file:
                    info_db = json.load(file)

                info = TASK_ENV.play_once()
                info_db[f"episode_{episode_idx}"] = info

                with open(info_file_path, "w", encoding="utf-8") as file:
                    json.dump(info_db, file, ensure_ascii=False, indent=4)

                TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
                TASK_ENV.merge_pkl_to_hdf5_video()
                TASK_ENV.remove_data_cache()
                assert TASK_ENV.check_success(), "Collect Error"
            except Exception as exc:
                if not args.get("success_first"):
                    raise
                try:
                    TASK_ENV.close_env()
                except Exception:
                    pass
                try:
                    TASK_ENV.remove_data_cache()
                except Exception:
                    pass
                failed_seed = seed_list[episode_idx]
                print(
                    f"materialization failed for episode {episode_idx}, seed={failed_seed}; "
                    "recording failure and advancing candidate stream"
                )
                reject_success_first_materialization(
                    args, seed_list, episode_idx, failed_seed, exc
                )
                # Resume through the same success-first entry. It will retain
                # completed HDF5 episodes, add one later candidate, and restart
                # materialization at this now-missing episode index.
                return run(TASK_ENV, args)

        command = f"cd description && bash gen_episode_instructions.sh {args['task_name']} {args['task_config']} {args['language_num']}"
        os.system(command)


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser = parser.parse_args()
    task_name = parser.task_name
    task_config = parser.task_config

    main(task_name=task_name, task_config=task_config)
