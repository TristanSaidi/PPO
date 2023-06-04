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
        self.episode_rewards = AverageScalarMeter(100)
        self.episode_lengths = AverageScalarMeter(100)
        self.obs = None
        self.epoch_num = 0

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

    def set_ppo_rollout_params(self, ppo_config):
        self.horizon_length = ppo_config["horizon_length"]
        self.batch_size = self.horizon_length * self.num_actors
        self.minibatch_size = ppo_config["minibatch_size"]
        self.mini_epochs_num = ppo_config["mini_epochs"]
        assert self.batch_size % self.minibatch_size == 0

    def configure_saving(self, ppo_config):
        self.save_freq = ppo_config["save_frequency"]
        self.save_best_after = ppo_config["save_best_after"]

    def configure_tensorboard(self):
        self.extra_info = {}
        self.writer = SummaryWriter(self.output_tb)

    def clear_stats(self):
        pass

    def train(self):
        pass

    def train_epoch(self):
        pass

    def get_full_state_weights(self):
        pass

    def set_full_state_weights(self, weights):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass

