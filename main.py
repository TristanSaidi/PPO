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
import yaml
import random

from algo.ppo import PPO


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
        obs = torch.tensor(obs, device=ppo_config["device"])
        action = policy.model.test_act(obs).detach().cpu().numpy()
        obs, rew, done, _, _ = env.step(action)
    
    env.close()

def train_agent(env, ppo_config):
    policy = PPO(env, ppo_config)
    policy.train()

def main()->int:

    parser =  argparse.ArgumentParser(
        description='Simulate agent in a gym environment'
    )

    parser.add_argument(
        '--test',
        type = int
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
    with open('configs/ppo.yaml') as f:
        ppo_config = yaml.safe_load(f)

    ppo_config["checkpoint"] = command_configs["checkpoint"]
    ppo_config["seed"] = set_seed(ppo_config["seed"])
    ppo_config["test"] = bool(command_configs["test"])
    ppo_config["checkpoint"] = command_configs["checkpoint"]
    ppo_config["experiment"] = command_configs["experiment"]

    if command_configs["test"] == True:
        env = gym.make(command_configs['env'], render_mode='human')
        test_agent(env, ppo_config)
    else:
        env = gym.make(command_configs['env'])
        train_agent(env, ppo_config)

    return 0

if __name__ ==  '__main__':
    sys.exit(main())