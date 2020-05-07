'''
PyTorch implementation of a Deep Q Network for use by Deep Q agent and
Neural Fictitious Self Play Agent.


@author : mukundv
'''

import torch

from functools import reduce
from torch import nn


'''
Fully connected Q network in PyTorch

Parameters:

    state_shape (list of int) : shape of state

    num_actions (int) : number of possible actions this agent can take

    hidden_layers (list) : list of hidden layer sizes to use in fully connected network

    activation (str) : which activation functions to use? 'tanh' or 'relu'
'''
class DQNet(nn.Module):

    def __init__(self, state_shape, num_actions, hidden_layers, activation='tanh'):
        super(DQNet, self).__init__()

        flattened_state_size = reduce(lambda x, y : x * y, state_shape)

        neurons = [flattened_state_size] + hidden_layers + [num_actions]
        layers = []

        # experiment with batch_norm, dropout

        # activations on all layers but the last
        for i in range(len(neurons) - 2):
            layers.append(nn.Linear(neurons[i], neurons[i+1]))
            if activation == 'relu':
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Tanh())

        layers.append(nn.Linear(neurons[-2], neurons[-1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, state):
        q_values = self.layers(state)
        return q_values
