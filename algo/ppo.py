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

from utils.misc import AdaptiveScheduler, AverageScalarMeter
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
        self.latest_episode_rewards = 0
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

        # if mode == test, load provided checkpoint
        if config["test"] == True:
            checkpoint = torch.load(config["checkpoint"])
            self.model.load_state_dict(checkpoint["model"])

    def configure_output_dir(self, output_dir):
        self.output_nn = os.path.join("runs/" + output_dir, "nn")
        self.output_tb = os.path.join("runs/" + output_dir, "tb")
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
        self.best_rewards = -1e10

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
            print(f"Iteration: {self.iter_num}")
            episode_rewards = self.latest_episode_rewards
            if self.iter_num % self.save_freq == 0:
                checkpoint = f'ep_{self.iter_num}_reward_{episode_rewards}'
                self.save_checkpoint(os.path.join(self.output_nn,checkpoint))
            if episode_rewards > self.best_rewards and self.iter_num > self.save_best_after:
                print(f"New Best Episode Reward: {episode_rewards}")
                best = os.path.join(self.output_nn, 'best')
                self.save_checkpoint(best)
                self.best_rewards = episode_rewards

    def save_checkpoint(self, name):
        weights = {
            'model' : self.model.state_dict(),
        }
        torch.save(weights, f'{name}.pth')

    def update_policy(self):

        rollout_dict = self.rollout()
        rollout_observations = rollout_dict["rollout_observations"]
        rollout_values = rollout_dict["rollout_values"]
        rollout_actions = rollout_dict["rollout_actions"]
        rollout_log_probabilities = rollout_dict["rollout_log_probabilities"]
        rollout_rtgs = rollout_dict["rollout_rtgs"]

        # compute advantage
        A = rollout_rtgs - rollout_values.detach()
        if self.normalize_advantage:
            A = (A - A.mean()) / (A.std() + 1e-10)

        for k in range(self.mini_epochs):

            values, action_log_probs = self.model.evaluate(
                rollout_observations,
                rollout_actions
            )
            action_log_probs = action_log_probs.to(self.device)
            # calculate r_theta
            r_theta = torch.exp(action_log_probs - rollout_log_probabilities)

            # calculate surrogate loss 1
            surrogate_loss_1 = A * r_theta
            # calculate surrogate loss 2
            surrogate_loss_2 = A * torch.clamp(r_theta, 1 - self.e_clip, 1 + self.e_clip)

            actor_loss = -1 * torch.min(surrogate_loss_1, surrogate_loss_2).mean()
            critic_loss = ((values - rollout_rtgs) ** 2).mean()
            loss = actor_loss + 0.5 * critic_loss

            self.actor_loss = actor_loss.clone().detach()
            self.critic_loss = critic_loss.clone().detach()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log tensorboard
            self._logger()


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
                rollout_actions.append(action)
                rollout_log_probabilities.append(action_log_prob)

                # update episode length and reward
                episode_rewards.append(reward)
                episode_length += 1
                total_rollout_time += 1
            
            # update running average episode length and reward
            self.latest_episode_rewards = torch.tensor(episode_rewards, device=self.device).sum().item()
            self.episode_lengths.update(torch.tensor([episode_length], device=self.device))

            # add episode reward and value information to rollout buffers
            rollout_rewards.append(episode_rewards)
            rollout_values.append(episode_values)

        # compute reward-to-go for each visited state in the rollout
        rollout_rtgs = self.compute_rtgs(rollout_rewards)
        # flatten rollout values
        rollout_values = [val for ep_vals in rollout_values for val in ep_vals]
        # convert to torch tensors and store in dictionary
        rollout_dict = {
            "rollout_observations" : torch.stack(rollout_observations),
            "rollout_values" : torch.stack(rollout_values),
            "rollout_actions" : torch.stack(rollout_actions),
            "rollout_log_probabilities" : torch.stack(rollout_log_probabilities).unsqueeze(1),
            "rollout_rtgs" : torch.tensor(rollout_rtgs, device=self.device).unsqueeze(1)
        }
        
        return rollout_dict

    def compute_rtgs(self, rollout_rewards):

        rollout_rtgs = []
        for episode_rewards in reversed(rollout_rewards):
            discounted_future_reward = 0
            # iterate backwards through the episode, computing rtg for each timestep
            for reward_t in reversed(episode_rewards):
                discounted_future_reward = reward_t + self.gamma * discounted_future_reward
                rollout_rtgs.insert(0, discounted_future_reward)
        
        return rollout_rtgs

    def _logger(self):
        self.writer.add_scalar('episode_rewards/step', self.latest_episode_rewards, self.agent_steps)
        self.writer.add_scalar('actor_loss/step', self.actor_loss, self.agent_steps)
        self.writer.add_scalar('critic_loss/step', self.critic_loss, self.agent_steps)



    

