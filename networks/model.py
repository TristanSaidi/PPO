# --------------------------------------------------------
# Tristan Saidi's PPO Implementation
# Actor and Policy network base classes
# Adapted from Haozhi Qi's implementation of PPO
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ActorCritic(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        # unpack actor and critic network architecture params
        self.actor_units = kwargs.pop('actor_units')
        self.critic_units = kwargs.pop('critic_units')
        self.num_actions = kwargs.pop('num_actions')
        self.input_shape = kwargs.pop('input_shape')

        # instantiate actor network
        self.actor_network = MLP(self.actor_units, self.input_shape)
        self.actor_mu = nn.Linear(self.actor_units[-1], self.num_actions)
        self.actor_log_sigma = nn.Linear(self.actor_units[-1], self.num_actions)

        # instatiate critic network
        self.critic_network = MLP(self.critic_units, self.input_shape)
        self.critic_value = nn.Linear(self.critic_units[-1], 1)
    
    def _actor_critic(self, observation):
        """
            Queries both the action and value networks

            Parameters:
                observation - the observation at the current timestep
        
            Return:
                mu - mean of action distribution
                log_sigma - log variance of each dim of action
                value - predicted value of current state
        """

        # pass obs through V net to get state value
        value_embed = self.critic_network(observation)
        value = self.critic_value(value_embed)

        # pass obs through critic net to get mu, sigma
        actor_embed = self.actor_network(observation)
        mu = self.actor_mu(actor_embed)
        log_sigma = self.actor_log_sigma(actor_embed)

        return mu, log_sigma, value

    @torch.no_grad()
    def train_act(self, observation):
        # passes observation through actor network and adds noise
        mu, log_sigma, value = self._actor_critic(observation)
        sigma = torch.exp(log_sigma)

        action, action_log_prob, entropy = self.sample_action_dist(mu, sigma)
        
        actor_critic_dict = {
            "action" : action,
            "mu" : mu,
            "sigma" : sigma,
            "value" : value,
            "action_log_prob" : action_log_prob,
            "entropy" : entropy
        }
        return actor_critic_dict


    @torch.no_grad()
    def test_act(self, observation):
        # passes observation through actor network
        mu, _ , _ = self._actor_critic(observation)
        return mu

    def sample_action_dist(self, mu, sigma):
        # construct current action distribution
        current_action_dist = torch.distributions.Normal(mu, sigma)
        action = current_action_dist.sample()

        # calculate log prob of action
        action_log_prob = current_action_dist.log_prob(action).sum()

        # calculate entropy of current action dist
        entropy = current_action_dist.entropy().sum(dim=-1)

        return action, action_log_prob, entropy

    def evaluate(self, rollout_observations, rollout_actions):
        mu, log_sigma, values = self._actor_critic(rollout_observations)
        sigma = torch.exp(log_sigma) # construct diag covariance matrix

        log_probs = []
        for (action_i, mu_i, sigma_i) in zip(rollout_actions, mu, sigma):
            current_action_dist = torch.distributions.Normal(mu_i, sigma_i)
            log_probs.append(current_action_dist.log_prob(action_i))
        return values, torch.stack(log_probs)