'''
PyTorch implementation of a Deep Recurrent Q Network (DRQN) for use by Deep Recurrent Q agent and
Recurrent Neural Fictitious Self Play Agent.


@author : mukundv
'''

import torch

from functools import reduce
from torch import nn
from torch.nn import LSTM

'''
Recurrent q network in PyTorch

Parameters:

    state_shape (list of int) : shape of state

    num_actions (int) : number of possible actions this agent can take

    hidden_size (int) : size of hidden state of recurrent layer

    num_layers (int) : number of recurrent layers to use

    hidden_layers (list) : list of hidden layer sizes describing the fully connected
                           network from the hidden state to output

    activation (str) : which activation functions to use ? 'tanh' or 'relu'
'''
class DRQNet(nn.Module):

    def __init__(self, state_shape, num_actions, hidden_size, num_layers,  hidden_layers, activation='relu', device=None):
        super(DRQNet, self).__init__()

        # initialize lstm layers
        self.flattened_state_size = reduce(lambda x, y : x * y, state_shape)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_layers = LSTM(self.flattened_state_size, hidden_size, num_layers)
        self.device = device

        # initialize fully connected layers
        neurons = [hidden_size] + hidden_layers + [num_actions]
        fc_layers = []
        for i in range(len(neurons) - 2):
            fc_layers.append(nn.Linear(neurons[i], neurons[i+1]))
            if activation == 'relu':
                fc_layers.append(nn.ReLU())
            else:
                fc_layers.append(nn.Tanh())

        fc_layers.append(nn.Linear(neurons[-2], neurons[-1]))
        self.fc_layers = nn.Sequential(*fc_layers)
        self.init_hidden(1)

    def forward(self, state):
        x, (self.hidden, self.cell) = self.lstm_layers(state.view(-1, 1, self.flattened_state_size), (self.hidden, self.cell))
        q_values = self.fc_layers(x)
        return q_values

    def init_hidden(self, size):
        self.hidden = torch.zeros(self.num_layers, size, self.hidden_size).to(self.device)
        self.cell = torch.zeros(self.num_layers, size, self.hidden_size).to(self.device)
