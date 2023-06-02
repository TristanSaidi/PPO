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
        value = self.critic_network(observation)

        # pass obs through critic net to get mu, sigma
        actor_embed = self.actor_network(observation)
        mu = self.actor_mu(actor_embed)
        log_sigma = self.actor_log_sigma(actor_embed)

        return mu, log_sigma, value

    def forward(self, observation_dict):
        """
            Forward pass through the actor critic module, returning
            information required for gradient updates

			Parameters:
				observation_dict - observation dictionary at the 
                current timestep. Contains previous action and current
                observation

			Return:
                actor_critic_dict - dictionary containing information
                about the forward pass through the AC module
		"""
        observation = observation_dict['observation']
        previous_action = observation_dict['previous_action']

        # query actor and critic nets
        mu, log_sigma, value = self._actor_critic(observation)
        sigma = torch.exp(log_sigma)
        
        previous_action_neg_log_prob, entropy = self.query_action_dist_information(
            mu,
            sigma,
            previous_action
        )

        actor_critic_dict = {
            "mu" : mu,
            "sigma" : sigma,
            "value" : value,
            "previous_action_neg_log_prob" : previous_action_neg_log_prob,
            "entropy" : entropy
        }

        return actor_critic_dict

    @torch.no_grad()
    def train_act(self, observation):
        # passes observation through actor network and adds noise
        mu, log_sigma, value = self._actor_critic(observation)
        previous_action_neg_log_prob, entropy = self.query_action_dist_information(
            mu,
            sigma,
            previous_action
        )
        actor_critic_dict = {
            "mu" : mu,
            "sigma" : sigma,
            "value" : value,
            "previous_action_neg_log_prob" : previous_action_neg_log_prob,
            "entropy" : entropy
        }
        return actor_critic_dict


    @torch.no_grad()
    def test_act(self, observation):
        # passes observation through actor network
        mu, _ , _ = self._actor_critic(observation)
        return mu

    def query_action_dist_information(self, mu, sigma, previous_action):
        # construct current action distribution
        current_action_dist = torch.distributions.Normal(mu, sigma)

        # calculate negative log prob of previous action under
        # current action distribution 
        previous_action_neg_log_prob = -1 * current_action_dist.log_prob(previous_action).sum(1)

        # calculate entropy of current action dist
        entropy = current_action_dist.entropy().sum(dim=-1)

        return previous_action_neg_log_prob, entropy

    



