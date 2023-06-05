# --------------------------------------------------------
# Tristan Saidi's PPO Implementation
# Script to run agent in gym environment
# Adapted from https://github.com/ericyangyu/PPO-for-Beginners
# --------------------------------------------------------

import argparse
import gym
import os
import numpy as np
import sys
import torch
import hydra
import random
from omegaconf import DictConfig, OmegaConf

from algo.ppo import PPO

OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)


def init_actor_critic(env, units):
    input_shape = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    args = {
        "actor_units" : units,
        "critic_units" : units,
        "num_actions" : num_actions,
        "input_shape" : input_shape,
    }
    return ActorCritic(args)


# from https://github.com/HaozhiQi/hora
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return seed

def test_agent(env, ppo_config):
    
    policy = PPO(env, ppo_config)
    obs, _ = env.reset()

    while True:
        env.render()

        # Query deterministic action from policy and run it
        obs = torch.tensor(obs)
        action = policy.test_act(obs).detach().numpy()
        obs, rew, done, _, _ = env.step(action)
        # Sum all episodic rewards as we go along
    
    env.close()

def train_agent(env, ppo_config):
    policy = PPO(env, ppo_config)
    policy.train()

@hydra.main(config_name='ppo', config_path='configs')
def main(ppo_config: DictConfig)->int:

    parser =  argparse.ArgumentParser(
        'Simulate agent in a gym environment'
    )

    parser.add_argument(
        '--test',
        default = False,
        type = bool
    )

    parser.add_argument(
        '--checkpoint',
        default = '',
        type = str
    )

    parser.add_argument(
        '--experiment',
        default = 'ppo_agent',
        type = str
    )

    parser.add_argument(
        '--env',
        default = 'Pendulum-v1',
        type = str
    )

    command_args = parser.parse_args()
    command_configs = command_args.__dict__
    
    # Copy command args to ppo_config dict
    ppo_config = OmegaConf.to_container(ppo_config)
    ppo_config["checkpoint"] = command_configs["checkpoint"]
    ppo_config["seed"] = set_seed(ppo_config["seed"])
    ppo_config["test"] = command_configs["test"]
    ppo_config["checkpoint"] = command_configs["checkpoint"]
    ppo_config["experiment"] = command_configs["experiment"]


    if command_configs["test"]:
        env = gym.make(command_configs['env'], render_mode='human')
        test_agent(env, ppo_config)
    else:
        env = gym.make(command_configs['env'])
        train_agent(env, ppo_config)

    return 0

if __name__ ==  '__main__':
    main()