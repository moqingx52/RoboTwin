import importlib
import os
import gc
import sys

import cv2
import torch
import yaml
from envs import *

sys.path.append("../../")

import logging
import multiprocessing as mp
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import gymnasium as gym
import numpy as np
from description.utils.generate_episode_instructions import (
    generate_episode_descriptions,
)
from envs._GLOBAL_CONFIGS import *
from envs.utils.create_actor import UnStableError


LOG_LEVEL = os.getenv("VECTOR_ENV_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.WARNING),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("concurrent.futures").setLevel(logging.WARNING)
logging.getLogger("curobo").setLevel(logging.ERROR)
INIT_DEBUG = os.getenv("VECTOR_ENV_INIT_DEBUG", "0") == "1"
DEBUG_LEVEL = int(os.getenv("VECTOR_ENV_DEBUG_LEVEL", "0"))
DEBUG_EVERY = max(1, int(os.getenv("VECTOR_ENV_DEBUG_EVERY", "1")))


def _init_dbg(msg: str):
    if INIT_DEBUG:
        print(f"[vecdbg] {msg}", flush=True)


def _debug_enabled(level: int) -> bool:
    return DEBUG_LEVEL >= level


def _arr_stats(x):
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return {"size": 0}
    return {
        "shape": list(arr.shape),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def update_obs(observation):
    obs_dict = observation.get("observation", {})
    # Allow qpos-only or camera-disabled configs to pass through without KeyError.
    full_image = obs_dict.get("head_camera", {}).get("rgb", None)
    left_wrist_image = (
        obs_dict.get("left_camera", {}).get("rgb", None)
    )
    right_wrist_image = (
        obs_dict.get("right_camera", {}).get("rgb", None)
    )
    state = observation["joint_action"]["vector"]

    return {
        "full_image": full_image,
        "left_wrist_image": left_wrist_image,
        "right_wrist_image": right_wrist_image,
        "state": state,
    }


class SubEnv:
    def __init__(
        self,
        env_id: int,
        task_name: str,
        args: dict,
        env_seed: int = None,
        instruction_type = "seen",
        global_lock=None,
    ):
        _init_dbg(f"SubEnv[{env_id}] __init__ start task_name={task_name}")
        self.env_id = env_id
        self.task_name = task_name
        self.args = args
        self.env_seed = env_seed
        if self.env_seed is None:
            self.env_seed = self.env_id
        self.instruction = None
        self.task = None
        _init_dbg(f"SubEnv[{env_id}] task slot ready (initialized in setup_task)")
        self.instruction_type = instruction_type
        self.global_lock = global_lock
        self.lock = threading.Lock()
        self.debug_step_count = 0
        _init_dbg(f"SubEnv[{env_id}] __init__ done seed={self.env_seed}")

    def setup_task(self):
        _init_dbg(f"SubEnv[{self.env_id}] setup_task start")
        self.close()
        self.task = None

        with self.global_lock:
            with self.lock:
                trial_seed = self.env_seed
                is_valid = False
                episode_info = {}
                while not is_valid:
                    task = None
                    try:
                        _init_dbg(
                            f"SubEnv[{self.env_id}] setup_demo try seed={trial_seed}"
                        )
                        task = class_decorator(self.task_name)
                        task.setup_demo(
                            now_ep_num=trial_seed,
                            seed=trial_seed,
                            is_test=True,
                            **self.args,
                        )
                        get_info = getattr(task, "get_info", None)
                        episode_info = get_info() if callable(get_info) else {}
                        is_valid = True
                        self.task = task
                        _init_dbg(
                            f"SubEnv[{self.env_id}] setup_demo success seed={trial_seed}"
                        )
                    except Exception as e:
                        _init_dbg(
                            f"SubEnv[{self.env_id}] setup_demo failed seed={trial_seed} err={type(e).__name__}: {e}"
                        )
                        if task is not None:
                            task.close_env()
                        trial_seed += 1
                        continue

                self.episode_info_list = [episode_info]
        _init_dbg(f"SubEnv[{self.env_id}] setup_task done")

    def create_instruction(self):
        task_descriptions = generate_episode_descriptions(
            self.task_name, self.episode_info_list, 1
        )
        instruction = np.random.choice(task_descriptions[0][self.instruction_type])
        return instruction

    def step(self, actions):
        if self.get_instruction() is None:
            self.reset(env_seed=None)

        with self.lock:
            self.debug_step_count += 1
            action_np = np.asarray(actions, dtype=np.float32)
            reward, termination, truncation, info = self.task.gen_sparse_reward_data(actions)
            obs = update_obs(self.task.get_obs())
            obs["instruction"] = self.task.get_instruction()
            if _debug_enabled(1) and (self.debug_step_count % DEBUG_EVERY == 0):
                if not isinstance(info, dict):
                    info = {"raw_info": info}
                dbg = {
                    "env_id": int(self.env_id),
                    "step_i": int(self.debug_step_count),
                    "reward": float(reward),
                    "terminated": bool(termination),
                    "truncated": bool(truncation),
                    "action_stats": _arr_stats(action_np),
                }
                run_steps = getattr(self.task, "run_steps", None)
                reward_step = getattr(self.task, "reward_step", None)
                if run_steps is not None:
                    dbg["run_steps"] = int(run_steps)
                if reward_step is not None:
                    dbg["reward_step"] = int(reward_step)
                plan_success = getattr(self.task, "plan_success", None)
                if plan_success is not None:
                    dbg["plan_success"] = bool(plan_success)
                if _debug_enabled(2):
                    state = obs.get("state")
                    if state is not None:
                        dbg["state_stats"] = _arr_stats(state)
                    success_keys = ("success", "success_once", "is_success", "done_success")
                    reward_keys = ("reward", "sparse_reward", "dense_reward", "reward_components")
                    info_focus = {}
                    for k, v in info.items():
                        if k in success_keys or k in reward_keys or ("success" in str(k).lower()):
                            info_focus[k] = v
                    if info_focus:
                        dbg["info_focus"] = info_focus
                if _debug_enabled(3):
                    flat = action_np.reshape(-1)
                    dbg["action_head"] = flat[: min(16, flat.size)].tolist()
                info["debug_trace"] = dbg

        return {
            "obs": obs,
            "reward": reward,
            "terminated": termination,
            "truncated": truncation,
            "info": info,
        }

    def reset(self, env_seed=None):
        with self.global_lock:
            with self.lock:
                if self.task is not None:
                    self.task.close_env()
                if env_seed is not None:
                    self.env_seed = env_seed

                self.instruction = self.create_instruction()
                self.args["instruction"] = self.instruction

                trial_seed = self.env_seed
                is_valid = False
                while not is_valid:
                    try:
                        self.task.setup_demo(
                            now_ep_num=trial_seed, seed=trial_seed, **self.args
                        )
                        self.task.step_lim = self.args["step_lim"]
                        self.task.run_steps = 0
                        self.task.reward_step = 0
                        is_valid = True
                    except UnStableError as e:
                        logging.warning(
                            f"RoboTwin SubEnv {self.env_id} reset error with seed {trial_seed}, error: {e}, trying new seed: {trial_seed + 1}"
                        )
                        self.task.close_env()
                        trial_seed += 1
                        continue
                    except Exception as e:
                        logging.error(
                            f"RoboTwin SubEnv {self.env_id} reset error with seed {trial_seed}, error: {e}"
                        )
                        self.task.close_env()
                        raise

        return

    def get_obs(self):
        with self.lock:
            obs = self.task.get_obs()
            obs = update_obs(obs)
            obs["instruction"] = self.task.get_instruction()

        return obs

    def get_instruction(self):
        with self.lock:
            if self.task is None:
                return None
            return self.instruction

    def close(self, clear_cache=True):
        if self.task is not None:
            with self.lock:
                self.task.close_env(clear_cache=clear_cache)

    def check_seed(self, seed):
        setup_demo_success = False
        play_once_success = False

        t1 = time.time()
        with self.global_lock:
            with self.lock:
                try:
                    self.task.setup_demo(now_ep_num=seed, seed=seed, **self.args)
                    setup_demo_success = True
                    _ = self.task.get_obs()
                    _ = self.task.play_once()
                    if self.task.plan_success and self.task.check_success():
                        play_once_success = True
                except Exception as e:
                    logging.warning(
                        f"RoboTwin SubEnv {self.env_id} check_seed error with seed {seed}, error: {e}"
                    )
        t2 = time.time()
        result = {
            "setup_demo_success": setup_demo_success,
            "play_once_success": play_once_success,
            "cost_time": t2 - t1,
        }
        return result


class VectorEnv(gym.Env):
    def __init__(
        self,
        task_config,
        n_envs,
        env_seeds=None,
        instruction_type="seen",
    ):
        _init_dbg(f"VectorEnv.__init__ enter n_envs={n_envs}")
        self.env_seeds = env_seeds
        if self.env_seeds is not None:
            assert len(self.env_seeds) == n_envs
        _init_dbg("VectorEnv.__init__ seeds checked")
        assets_path = os.getenv("ASSETS_PATH")
        self.task_name = task_config.get("task_name")

        head_camera_type = "D435"
        rdt_step = 10
        args = task_config

        args["planner_backend"] = args.get("planner_backend", "curobo") # Choices: [curobo, mplib]
        _init_dbg(
            f"VectorEnv.__init__ task={self.task_name} planner_backend={args['planner_backend']}"
        )

        embodiment_type = args.get("embodiment")
        embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

        with open(embodiment_config_path, "r", encoding="utf-8") as f:
            _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
        _init_dbg("VectorEnv.__init__ loaded embodiment config")

        with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
            _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        _init_dbg("VectorEnv.__init__ loaded camera config")

        args["head_camera_h"] = _camera_config[head_camera_type]["h"]
        args["head_camera_w"] = _camera_config[head_camera_type]["w"]

        def get_embodiment_file(embodiment_type):
            robot_file = _embodiment_types[embodiment_type]["file_path"]
            if robot_file is None:
                raise "No embodiment files"
            return robot_file

        def get_embodiment_config(robot_file):
            robot_config_file = os.path.join(robot_file, "config.yml")
            with open(robot_config_file, "r", encoding="utf-8") as f:
                embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
            return embodiment_args

        if len(embodiment_type) == 1:
            args["left_robot_file"] = os.path.join(
                assets_path, get_embodiment_file(embodiment_type[0])
            )
            args["right_robot_file"] = os.path.join(
                assets_path, get_embodiment_file(embodiment_type[0])
            )
            args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            args["left_robot_file"] = os.path.join(
                assets_path, get_embodiment_file(embodiment_type[0])
            )
            args["right_robot_file"] = os.path.join(
                assets_path, get_embodiment_file(embodiment_type[1])
            )
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False
        else:
            raise "embodiment items should be 1 or 3"
        _init_dbg("VectorEnv.__init__ embodiment files resolved")

        args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
        args["right_embodiment_config"] = get_embodiment_config(
            args["right_robot_file"]
        )

        if len(embodiment_type) == 1:
            embodiment_name = str(embodiment_type[0])
        else:
            embodiment_name = str(embodiment_type[0]) + "_" + str(embodiment_type[1])

        args["embodiment_name"] = embodiment_name

        args["rdt_step"] = rdt_step
        args["save_path"] += f"/{args['task_name']}_reward"

        args["n_envs"] = n_envs
        args["action_dim"] = 14

        args["eval_mode"] = True
        args["eval_video_log"] = False
        args["render_freq"] = 0

        self.args = args
        self.n_envs = n_envs

        self.envs = []
        self.instruction_type = instruction_type

        self.global_lock = threading.Lock()

        self.env_thread_pool = ThreadPoolExecutor(max_workers=n_envs)
        _init_dbg("VectorEnv.__init__ thread pool created")

        self._init_envs()
        _init_dbg("VectorEnv.__init__ done")

    def _init_envs(self):
        _init_dbg("VectorEnv._init_envs start")
        for i in range(self.n_envs):
            _init_dbg(f"VectorEnv._init_envs building SubEnv[{i}]")
            sub_env = SubEnv(
                env_id=i,
                task_name=self.task_name,
                args=self.args,
                env_seed=self.env_seeds[i] if self.env_seeds else None,
                instruction_type=self.instruction_type,
                global_lock=self.global_lock,
            )
            sub_env.setup_task()
            self.envs.append(sub_env)
            _init_dbg(f"VectorEnv._init_envs SubEnv[{i}] ready")
        _init_dbg("VectorEnv._init_envs done")

    def transform(self, results):
        res_dict = defaultdict(list)
        for res in results:
            for k, v in res.items():
                res_dict[k].append(v)

        res_dict = dict(res_dict)
        return (
            res_dict["obs"],
            res_dict["reward"],
            res_dict["terminated"],
            res_dict["truncated"],
            res_dict["info"],
        )

    def step(self, actions):
        if len(self.envs) == 0:
            self._init_envs()

        step_futures = {}
        for i in range(self.n_envs):
            future = self.env_thread_pool.submit(self.envs[i].step, actions[i])
            step_futures[i] = future

        results = []
        for i in range(self.n_envs):
            future = step_futures[i]
            try:
                result = future.result(timeout=120)
                results.append(result)
            except Exception as e:
                raise RuntimeError(f"SubEnv {i} step error: {e}")

        obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
            self.transform(results)
        )

        return obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv

    def reset(self, env_idx=None, env_seeds=None):
        if len(self.envs) == 0:
            self._init_envs()

        if env_idx is None:
            env_idx = list(range(self.n_envs))
        elif isinstance(env_idx, (list, tuple)):
            env_idx = list(env_idx)
        elif isinstance(env_idx, torch.Tensor):
            env_idx = env_idx.tolist()
        else:
            env_idx = [env_idx]

        reset_futures = {}
        for idx in env_idx:
            if 0 <= idx < self.n_envs:
                seed = None
                if env_seeds is not None and len(env_seeds) == len(env_idx):
                    seed_idx = env_idx.index(idx)
                    seed = env_seeds[seed_idx]

                future = self.env_thread_pool.submit(
                    self.envs[idx].reset, env_seed=seed
                )
                reset_futures[idx] = future

        for idx in env_idx:
            if 0 <= idx < self.n_envs:
                future = reset_futures[idx]
                try:
                    future.result(timeout=120)
                except Exception as e:
                    raise RuntimeError(f"SubEnv {idx} reset error: {e}")

    def get_obs(self):
        obs_venv = []
        for env in self.envs:
            obs_venv.append(env.get_obs())

        return obs_venv

    def close(self, clear_cache=True):
        for env in self.envs:
            env.close(clear_cache=clear_cache)

        if clear_cache:
            for env in self.envs:
                env = None
            self.envs = []
            gc.collect()
            torch.cuda.empty_cache()

    def check_seeds(self, seeds: list[int]):
        assert len(seeds) == self.n_envs
        check_futures = {}
        for i in range(self.n_envs):
            future = self.env_thread_pool.submit(self.envs[i].check_seed, seeds[i])
            check_futures[i] = future

        results = [None] * self.n_envs
        for future in as_completed(check_futures.values(), timeout=120):
            for idx, f in check_futures.items():
                if f == future:
                    try:
                        result = future.result()
                        results[idx] = result
                    except Exception as e:
                        raise RuntimeError(f"SubEnv {idx} check seed error: {e}")
                    break

        return results


if __name__ == "__main__":
    mp.set_start_method("spawn")  # solve CUDA compatibility problem
    task_name = "place_shoe"
    n_envs = 4
    steps = 30
    horizon = 10
    action_dim = 14
    times = 10
    env = VectorEnv(task_name, n_envs, horizon)
    actions = np.zeros((n_envs, horizon, action_dim))
    for t in range(times):
        prev_obs_venv, reward_venv, truncation, termination, info_venv = (
            env.reset()
        )
        for step in range(steps):
            actions += np.random.randn(n_envs, horizon, action_dim) * 0.05
            actions = np.clip(actions, 0, 1)
            obs_venv, reward_venv, truncation, termination, info_venv = env.step(
                actions
            )

            # 测试partial reset功能
            if step % 10 == 0:
                # 重置所有环境
                env.reset()
            elif step % 5 == 0:
                # 只重置环境0和2
                env.reset(env_idx=[0, 2])
            
            obs = (
                env.get_obs()
            )
        env.close()