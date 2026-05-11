# import packages and module here
import sys, os
import yaml
from .model import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def _resolve_rdt_base_config(model_name, usr_args):
    requested_config = usr_args.get("rdt_base_config")
    if requested_config and requested_config != "auto":
        return requested_config

    model_config_path = os.path.join(parent_directory, "model_config", f"{model_name}.yml")
    pretrained_model = ""
    if os.path.isfile(model_config_path):
        with open(model_config_path, "r", encoding="utf-8") as f:
            model_config = yaml.safe_load(f) or {}
        if model_config.get("rdt_base_config"):
            return model_config["rdt_base_config"]
        pretrained_model = str(model_config.get("pretrained_model_name_or_path", ""))

    model_hint = f"{model_name} {pretrained_model}".lower()
    if "170m" in model_hint or "rdt-170m" in model_hint:
        return "configs/base_170m.yaml"
    return "configs/base.yaml"


def encode_obs(observation):  # Post-Process Observation
    observation["agent_pos"] = observation["joint_action"]["vector"]
    return observation


def get_model(usr_args):  # keep
    model_name = usr_args["ckpt_setting"]
    checkpoint_id = usr_args["checkpoint_id"]
    left_arm_dim, right_arm_dim, rdt_step = (
        usr_args["left_arm_dim"],
        usr_args["right_arm_dim"],
        usr_args["rdt_step"],
    )
    rdt_base_config = _resolve_rdt_base_config(model_name, usr_args)
    print(f"[RDT] Using base config: {rdt_base_config}")
    rdt = RDT(
        os.path.join(
            parent_directory,
            f"checkpoints/{model_name}/checkpoint-{checkpoint_id}/pytorch_model/mp_rank_00_model_states.pt",
        ),
        usr_args["task_name"],
        left_arm_dim,
        right_arm_dim,
        rdt_step,
        rdt_base_config=rdt_base_config,
    )
    return rdt


def eval(TASK_ENV, model, observation):
    """x
    All the function interfaces below are just examples
    You can modify them according to your implementation
    But we strongly recommend keeping the code logic unchanged
    """
    obs = encode_obs(observation)  # Post-Process Observation
    instruction = TASK_ENV.get_instruction()
    input_rgb_arr, input_state = [
        obs["observation"]["head_camera"]["rgb"],
        obs["observation"]["right_camera"]["rgb"],
        obs["observation"]["left_camera"]["rgb"],
    ], obs["agent_pos"]  # TODO

    if (model.observation_window
            is None):  # Force an update of the observation at the first frame to avoid an empty observation window
        model.set_language_instruction(instruction)
        model.update_observation_window(input_rgb_arr, input_state)

    actions = model.get_action()[:model.rdt_step, :]  # Get Action according to observation chunk

    for action in actions:  # Execute each step of the action
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        input_rgb_arr, input_state = [
            obs["observation"]["head_camera"]["rgb"],
            obs["observation"]["right_camera"]["rgb"],
            obs["observation"]["left_camera"]["rgb"],
        ], obs["agent_pos"]  # TODO
        model.update_observation_window(input_rgb_arr, input_state)  # Update Observation


def reset_model(
        model):  # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.reset_obsrvationwindows()
