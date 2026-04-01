import yaml

from .dp_model import DP
from .rlinf_adapter import encode_observation_dict as encode_obs


def get_model(usr_args):
    ckpt_file = f"./policy/DP/checkpoints/{usr_args['task_name']}-{usr_args['ckpt_setting']}-{usr_args['expert_data_num']}-{usr_args['seed']}/{usr_args['checkpoint_num']}.ckpt"
    action_dim = usr_args['left_arm_dim'] + usr_args['right_arm_dim'] + 2 # 2 gripper
    
    load_config_path = f'./policy/DP/diffusion_policy/config/robot_dp_{action_dim}.yaml'
    with open(load_config_path, "r", encoding="utf-8") as f:
        model_training_config = yaml.safe_load(f)
    
    n_obs_steps = model_training_config['n_obs_steps']
    n_action_steps = model_training_config['n_action_steps']
    
    return DP(ckpt_file, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)


def eval(TASK_ENV, model, observation):
    """
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment
    """
    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()

    # ======== Get Action ========
    actions = model.get_action(obs)

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)

def reset_model(model):
    model.reset_obs()
