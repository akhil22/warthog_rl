#!/usr/bin/env python
import gym
from env.WarthogEnv import WarthogEnv
import time
import numpy
import torch
from torch import nn
from torch.distributions import Normal
import matplotlib.pyplot as plt
import rospy
from nav_msgs import Path

class PolicyNetworkGauss(nn.Module):
    def __init__(self, obs_dimension, sizes, action_dimension, act=nn.ReLU):
        super(PolicyNetworkGauss, self).__init__()
        sizes = [obs_dimension] + sizes + [action_dimension]
        out_activation = nn.Identity
        self.layers = []
        for j in range(0, len(sizes) - 1):
            act_l = act if j < len(sizes) - 2 else out_activation
            self.layers += [nn.Linear(sizes[j], sizes[j + 1]), act_l()]
        self.mu = nn.Sequential(*self.layers)
        log_std = -0.5 * np.ones(action_dimension, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def forward(self, x):
        mean = self.mu(x)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        return dist

