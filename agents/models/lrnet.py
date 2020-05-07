'''
PyTorch implementation of a Logistic Regression Approximation for Q-learning values (DRQN) for use by
Logistic Regression Q agent.


@author : mukundv
'''

import torch

from functools import reduce
from torch import nn


'''
Logistic Regression implementation in PyTorch

Parameters:

    state_shape (list of int) : shape of state

    num_actions (int) : number of possible actions this agent can take
'''
class LRQNet(nn.Module):

    def __init__(self, state_shape, num_actions):
        super(LRQNet, self).__init__()

        flattened_state_size = reduce(lambda x, y : x * y, state_shape)

        # experiment with batch_norm, dropout

        self.layer = nn.Linear(flattened_state_size, num_actions)

    def forward(self, state):
        q_values = self.layer(state)
        return q_values
