import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HillClimbingPolicy(nn.Module):
    """
    Simple neural network policy for solving the hill climbing task.
    Consists of one common dense layer for both policy and value estimate and
    another dense layer for each.
    """

    def __init__(self, n_obs, n_hidden, n_actions):
        super(HillClimbingPolicy, self).__init__()

        self.n_obs = n_obs
        self.n_hidden = n_hidden
        self.n_actions = n_actions

        self.dense1 = nn.Linear(n_obs, n_hidden)
        self.dense_p = nn.Linear(n_hidden, n_actions)
        self.dense_v = nn.Linear(n_hidden, 1)

    def forward(self, obs):
        obs_one_hot = torch.zeros((obs.shape[0], self.n_obs))
        obs_one_hot[np.arange(obs.shape[0]), obs] = 1.0
        h_relu = F.relu(self.dense1(obs_one_hot))
        logits = self.dense_p(h_relu)
        policy = F.softmax(logits, dim=1)

        value = self.dense_v(h_relu).view(-1)

        return logits, policy, value

    def step(self, obs):
        """
        Returns policy and value estimates for given observations.
        :param obs: Array of shape [N] containing N observations.
        :return: Policy estimate [N, n_actions] and value estimate [N] for
        the given observations.
        """
        obs = torch.from_numpy(obs)
        _, pi, v = self.forward(obs)

        return pi.detach().numpy(), v.detach().numpy()
