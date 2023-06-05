# --------------------------------------------------------
# Tristan Saidi's PPO Implementation
# PPO base class
# Adapted from Haozhi Qi's implementation of PPO
# --------------------------------------------------------

import copy
import os
import numpy as np
import time
import gym
import torch
from tensorboardX import SummaryWriter

from networks.running_mean_std import RunningMeanStd
from utils.misc import AdaptiveScheduler, AverageScalarMeter
from utils.replay_buffer import ReplayBuffer
from networks.model import ActorCritic

class PPO(object):
    def __init__(self, env, config):
        self.device = config["device"]
        self.build_environment(env, config)
        self.init_model(config)
        self.configure_output_dir(config["experiment"])
        self.configure_optimizer(config["ppo"])
        self.configure_scheduler(config["ppo"])
        self.set_ppo_train_params(config["ppo"])
        self.set_ppo_rollout_params(config["ppo"])
        self.configure_saving(config["ppo"])
        self.configure_tensorboard()
        self.episode_rewards = AverageScalarMeter(100)
        self.episode_lengths = AverageScalarMeter(100)
        self.iter_num = 0

    def build_environment(self, env, config):
        self.env = env
        self.num_actors = config["ppo"]["num_actors"]
        action_space = self.env.action_space
        self.actions_num = action_space.shape[0]
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.device)
        self.observation_space = self.env.observation_space
        self.obs_shape = self.observation_space.shape[0]

    def init_model(self, config):
        args = {
            "actor_units" : config["network"]["mlp"]["units"],
            "critic_units" : config["network"]["mlp"]["units"],
            "num_actions" : self.actions_num,
            "input_shape" : self.obs_shape,
        }
        self.model = ActorCritic(args)
        self.model.to(self.device)
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)

    def configure_output_dir(self, output_dir):
        self.output_nn = os.path.join(output_dir, "nn")
        self.output_tb = os.path.join(output_dir, "tb")
        os.makedirs(self.output_nn, exist_ok = True)
        os.makedirs(self.output_tb, exist_ok = True)

    def configure_optimizer(self, ppo_config):
        self.last_lr = float(ppo_config["learning_rate"])
        self.weight_decay = 0.0
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.last_lr, weight_decay=self.weight_decay)
        
    def configure_scheduler(self, ppo_config):
        self.kl_threshold = ppo_config["kl_threshold"]
        self.scheduler = AdaptiveScheduler(self.kl_threshold)
    
    def set_ppo_train_params(self, ppo_config):
        self.e_clip = ppo_config["e_clip"]
        self.clip_value = ppo_config["clip_value"]
        self.entropy_coef = ppo_config["entropy_coef"]
        self.critic_coef = ppo_config["critic_coef"]
        self.bounds_loss_coef = ppo_config["bounds_loss_coef"]
        self.gamma = ppo_config["gamma"]
        self.tau = ppo_config["tau"]
        self.truncate_grads = ppo_config["truncate_grads"]
        self.grad_norm = ppo_config["grad_norm"]
        self.value_bootstrap = ppo_config["value_bootstrap"]
        self.normalize_advantage = ppo_config["normalize_advantage"]
        self.normalize_input = ppo_config["normalize_input"]
        self.normalize_value = ppo_config["normalize_value"]
        self.max_agent_steps = ppo_config["max_agent_steps"]
        self.max_episode_length = ppo_config["max_episode_steps"]

    def set_ppo_rollout_params(self, ppo_config):
        self.horizon_length = ppo_config["horizon_length"]
        self.batch_size = self.horizon_length * self.num_actors
        self.mini_epochs = ppo_config["mini_epochs"]

    def configure_saving(self, ppo_config):
        self.save_freq = ppo_config["save_frequency"]
        self.save_best_after = ppo_config["save_best_after"]

    def configure_tensorboard(self):
        self.extra_info = {}
        self.writer = SummaryWriter(self.output_tb)

    def train(self):
        """ 
        Main train loop for the PPO agent. This function calls
        train_iter() in a loop until the maximum number of steps
        is exceeded.
        """ 
        self.env.reset()
        self.agent_steps = self.batch_size

        while self.agent_steps < self.max_agent_steps:
            self.iter_num += 1
            self.agent_steps += self.batch_size
            self.update_policy()
            print(f"Average Episode Reward: {self.episode_rewards.get_mean()}")
            print(f"Iteration: {self.iter_num}")
            self._logger()


    def update_policy(self):

        rollout_dict = self.rollout()
        rollout_observations = rollout_dict["rollout_observations"]
        rollout_values = rollout_dict["rollout_values"]
        rollout_actions = rollout_dict["rollout_actions"]
        rollout_log_probabilities = rollout_dict["rollout_log_probabilities"]
        rollout_rtgs = rollout_dict["rollout_rtgs"]

        # compute advantage
        A = rollout_rtgs - rollout_values.detach()
        # if self.normalize_advantage:
        #     # normalize advantage
        #     # helps with training stability according to: https://github.com/ericyangyu/PPO-for-Beginners
        #     A = (A - A.mean()) / (A.std() + 1e-10)
        # perform optimization for mini_epochs
        for k in range(self.mini_epochs):

            action_log_probs = self.model.calculate_action_likelihood(
                rollout_observations,
                rollout_actions
            )

            # calculate r_theta
            r_theta = torch.exp(action_log_probs - rollout_log_probabilities)
            # calculate surrogate loss 1
            surrogate_loss_1 = A * r_theta
            # calculate surrogate loss 2
            surrogate_loss_2 = A * torch.clamp(r_theta, 1 - self.e_clip, 1 + self.e_clip)

            actor_loss = -1 * torch.min(surrogate_loss_1, surrogate_loss_2).mean()
            
            critic_loss = (rollout_values - rollout_rtgs) ** 2
            loss = actor_loss + 0.5 * critic_loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def rollout(self):
        """
        Rollout the agent for horizon_length number of steps.
        
        Returns: tuple containing info about the rollout
        """

        rollout_observations = []
        rollout_values = []
        rollout_actions = []
        rollout_log_probabilities = []
        rollout_rewards = []

        obs, _ = self.env.reset()
        done = False
        # keep track of episode len and reward to log averages
        total_rollout_time = 0

        # loop for entire rollout horizon
        while total_rollout_time <= self.horizon_length:
            
            episode_rewards = []
            episode_values = []
            episode_length = 0
            done = False

            # loop for single episode
            while not done and episode_length <= self.max_episode_length:
                # sample a_t from policy
                obs = torch.tensor(obs, device=self.device)
                actor_critic_dict = self.model.train_act(obs)

                # store o_t, v(o_t)
                rollout_observations.append(obs)
                episode_values.append(actor_critic_dict["value"])

                action = actor_critic_dict["action"]
                action_log_prob = actor_critic_dict["action_log_prob"]
                
                # apply a_t to environment
                obs, reward, done, info, _ = self.env.step(action.cpu().numpy())
                
                # store a_t, log_prob, r_t, v(o_t)
                rollout_actions.append(action.item())
                rollout_log_probabilities.append(action_log_prob)

                # update episode length and reward
                episode_rewards.append(reward)
                episode_length += 1
                total_rollout_time += 1
            
            # update running average episode length and reward
            self.episode_rewards.update(torch.tensor(episode_rewards, device=self.device).mean().unsqueeze(0))
            self.episode_lengths.update(torch.tensor([episode_length], device=self.device))

            # add episode reward and value information to rollout buffers
            rollout_rewards.append(episode_rewards)
            rollout_values.append(episode_values)

        # compute reward-to-go for each visited state in the rollout
        rollout_rtgs = self.compute_rtgs(rollout_rewards, rollout_values)
        # flatten rollout values
        rollout_values = [val for ep_vals in rollout_values for val in ep_vals]
        # convert to torch tensors and store in dictionary
        rollout_dict = {
            "rollout_observations" : torch.stack(rollout_observations),
            "rollout_values" : torch.stack(rollout_values),
            "rollout_actions" : torch.tensor(rollout_actions, dtype=torch.float, device=self.device),
            "rollout_log_probabilities" : torch.tensor(rollout_log_probabilities, dtype=torch.float, device=self.device),
            "rollout_rtgs" : torch.tensor(rollout_rtgs, device=self.device).unsqueeze(1)
        }
        
        return rollout_dict

    def compute_rtgs(self, rollout_rewards, rollout_values):

        rollout_rtgs = []
        for (episode_rewards, episode_values) in zip(reversed(rollout_rewards), reversed(rollout_values)):
            # start with value from final timestep of the episode
            discounted_future_reward = episode_values[-1]
            # iterate backwards through the episode, computing rtg for each timestep
            for reward_t in reversed(episode_rewards):
                discounted_future_reward = reward_t + self.gamma * discounted_future_reward
                rollout_rtgs.insert(0, discounted_future_reward)
        
        return rollout_rtgs

    def _logger(self):
        self.writer.add_scalar('episode_rewards/step', self.episode_rewards.get_mean(), self.agent_steps)



    

