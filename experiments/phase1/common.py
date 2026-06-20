import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(REPO_ROOT / "policy") not in sys.path:
    sys.path.append(str(REPO_ROOT / "policy"))
if str(REPO_ROOT / "description" / "utils") not in sys.path:
    sys.path.append(str(REPO_ROOT / "description" / "utils"))

TASKS = (
    "move_can_pot",
    "place_container_plate",
    "click_alarmclock",
    "dump_bin_bigbin",
)

VARIANTS = (
    "success",
    "seed_balanced",
    "difficulty_weighted",
)


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_task_args(task_name: str, task_config: str) -> Dict:
    from envs import CONFIGS_PATH

    args = load_yaml(repo_path("task_config", f"{task_config}.yml"))
    args["task_name"] = task_name
    args["task_config"] = task_config

    embodiment_config_path = Path(CONFIGS_PATH) / "_embodiment_config.yml"
    embodiment_types = load_yaml(embodiment_config_path)

    def get_embodiment_file(embodiment_type):
        robot_file = embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise RuntimeError(f"No embodiment file for {embodiment_type}")
        return robot_file

    def get_embodiment_config(robot_file):
        return load_yaml(Path(robot_file) / "config.yml")

    embodiment_type = args["embodiment"]
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
        raise RuntimeError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    camera_config = load_yaml(Path(CONFIGS_PATH) / "_camera_config.yml")
    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_camera_type]["h"]
    args["head_camera_w"] = camera_config[head_camera_type]["w"]
    args["render_freq"] = 0
    args["eval_mode"] = True
    return args


def make_task_env(task_name: str):
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def iter_jsonl(path: Path) -> Iterable[Dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_range(items: List[int], shard_id: int, num_shards: int) -> List[int]:
    if num_shards <= 1:
        return items
    return [item for i, item in enumerate(items) if i % num_shards == shard_id]


def load_hdf5_episode(path: Path):
    import h5py

    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as root:
        vector = root["/joint_action/vector"][()]
        image_dict = {
            cam_name: root[f"/observation/{cam_name}/rgb"][()]
            for cam_name in root["/observation/"].keys()
        }
    return vector, image_dict


def episode_to_arrays(path: Path):
    import cv2

    vector, image_dict = load_hdf5_episode(path)
    head_images = []
    states = []
    actions = []
    for j in range(vector.shape[0]):
        if j != vector.shape[0] - 1:
            head_img = cv2.imdecode(np.frombuffer(image_dict["head_camera"][j], np.uint8), cv2.IMREAD_COLOR)
            head_images.append(head_img)
            states.append(vector[j])
        if j != 0:
            actions.append(vector[j])
    return (
        np.moveaxis(np.asarray(head_images), -1, 1),
        np.asarray(states, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
    )


def copy_episode(src: Path, dst_dir: Path, episode_name: str) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{episode_name}.hdf5"
    shutil.copy2(src, dst)
    return dst


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", dest="task_name", required=True, choices=TASKS)
    parser.add_argument("--task-config", default="demo_clean")

