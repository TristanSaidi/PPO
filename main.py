# --------------------------------------------------------
# Tristan Saidi's PPO Implementation
# Script to run agent in gym environment
# Adapted from https://github.com/ericyangyu/PPO-for-Beginners
# --------------------------------------------------------

import argparse
import gym
import sys
import torch

# from ppo import PPO
from networks.model import ActorCritic

def test_agent(env, checkpoint):
    print("test")
    args = {
        "actor_units" : [3, 10, 3],
        "critic_units" : [3, 10, 3],
        "num_actions" : 1,
        "input_shape" : 3
    }
    policy = ActorCritic(args)

    obs, _ = env.reset()

    while True:
        env.render()

        # Query deterministic action from policy and run it
        obs = torch.tensor(obs)
        action = policy.test_act(obs).detach().numpy()
        obs, rew, done, _, _ = env.step(action)
        # Sum all episodic rewards as we go along
    
    env.close()

def train(env, checkpoint):
    pass

def main()->int:

    parser =  argparse.ArgumentParser(
        'Simulate agent in a gym environment'
    )

    parser.add_argument(
        '--test',
        default = True,
        type = bool
    )

    parser.add_argument(
        '--checkpoint',
        default = '',
        type = str
    )

    parser.add_argument(
        '--env',
        default = 'Pendulum-v1',
        type = str
    )

    args = parser.parse_args()
    configs = args.__dict__

    env = gym.make(configs['env'], render_mode='human')
    
    test_agent(env, configs["checkpoint"])

    return 0

if __name__ ==  '__main__':
    main()