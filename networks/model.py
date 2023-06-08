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
        self.device = kwargs.pop('device')

        # instantiate actor network
        self.actor_network = MLP(self.actor_units, self.input_shape)
        self.actor_mu = nn.Linear(self.actor_units[-1], self.num_actions)
		# Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.num_actions,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)
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

        return mu, self.cov_mat, value

    @torch.no_grad()
    def train_act(self, observation):
        """ Forward pass through actor critic module. Meant to 
        be called in a training related function, as action 
        includes added stochasticity.

        Args:
            observation (torch.tensor): environment observation

        Returns:
            dict: dictionary containing info about forward pass
        """
        mu, sigma, value = self._actor_critic(observation)

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
        """ Forward pass through actor critic module. Meant to
        be called during inference time.

        Args:
            observation (torch.tensor): environment observation

        Returns:
            torch.tensor: mean of action distribution
        """
        # passes observation through actor network
        mu, _ , _ = self._actor_critic(observation)
        return mu

    def sample_action_dist(self, mu, sigma):
        """ Creates and samples from action distribution based on
        provided parameters. Function returns sampled action,
         log probability of that action and the entropy
        of the dist

        Args:
            mu (torch.tensor): _description_
            sigma (torch.tensor): _description_

        Returns:
            tuple: sampled action, log probability of that action,
            entropy of action dist
        """
        # construct current action distribution
        current_action_dist = torch.distributions.MultivariateNormal(mu, sigma)
        action = current_action_dist.sample()

        # calculate log prob of action
        action_log_prob = current_action_dist.log_prob(action).sum()

        # calculate entropy of current action dist
        entropy = current_action_dist.entropy().sum(dim=-1)

        return action.detach(), action_log_prob.detach(), entropy

    def evaluate(self, rollout_observations, rollout_actions):
        """ Evaluates the values and log probabilities of visited states
        and actions (respectively) during rollout.

        Args:
            rollout_observations (torch.tensor): visited states during rollout
            rollout_actions (torch.tensor): actions taken during rollout

        Returns:
            tuple: values and log probabilities
        """
        mu, sigma, values = self._actor_critic(rollout_observations)
        dist = torch.distributions.MultivariateNormal(mu, sigma)
        log_probs = dist.log_prob(rollout_actions).unsqueeze(1)
        return values, log_probs