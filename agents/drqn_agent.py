'''
PyTorch implementation of Deep Recurrent Q-Learning agent for rlcard platform

@author : mukundv

Referenced and used util functions from:
https://github.com/datamllab/rlcard/blob/master/rlcard/agents/dqn_agent.py
https://github.com/datamllab/rlcard/blob/master/rlcard/utils/utils.py
'''

import numpy as np
import torch

from .models.drqnet import DRQNet
from rlcard.utils.utils import remove_illegal
from random import sample
from torch import nn




'''
deep recurrent q learning agent. uses agent wrapper as specified:
http://rlcard.org/development.html#developping-algorithms

Parameters:

    num_actions (int) : how many possible actions

    state_shape (list) : tensor shape of state

    hidden_size (int) : size of LSTM hidden state

    num_layers (int) : number of LSTM layers to use in our recurrent network

    hidden_layers (list) : hidden layer size

    memory_size (int) : max number of experiences to store in memory buffer

    copy_every (int) : how often to copy parameters to target network

    epsilons (list) : list of epsilon values to use over training period

    shift_epsilon_every (int) : how often should we shift our epsilon value

    gamma (float) : discount parameter

    device (torch.device) : device to put models on
'''
class DRQNAgent():

    def __init__(self,
                 num_actions,
                 state_shape,
                 hidden_size=128,
                 num_layers=2,
                 hidden_layers=[128,128],
                 lr = .0001,
                 batch_size = 32,
                 memory_size = 10000,
                 copy_every = 1000,
                 epsilons = [.1 * x for x in range(10, 0, -1)],
                 shift_epsilon_every = 1000,
                 gamma=.9,
                 device=None):

        self.num_actions = num_actions
        self.batch_size = batch_size
        self.copy_every = copy_every
        self.epsilons = epsilons
        self.shift_epsilon_every = shift_epsilon_every
        self.gamma = gamma
        self.device = device
        self.use_raw = False

        # initialize learner and target networks
        self.learner_net = DRQNet(state_shape, num_actions, hidden_size, num_layers, hidden_layers, 'relu', device).to(device)
        self.target_net = DRQNet(state_shape, num_actions, hidden_size, num_layers, hidden_layers, 'relu', device).to(device)
        self.learner_net.eval()
        self.target_net.eval()

        # initialize optimizer for learner network
        self.optim = torch.optim.Adam(self.learner_net.parameters(), lr=lr)

        # initialize loss function for network
        self.criterion = nn.MSELoss(reduction='mean').to(device)

        # how many timesteps have we seen?
        self.timestep = 0


        # initialize memory buffer
        self.memory_buffer = SeqMemory(memory_size, batch_size)

        self.device = device


    '''
    Given state, produce actions to generate training data. Use epsilon greedy action selection.
    Should be separate from compute graph as we only update through the feed function.

    Uses epsilon greedy methods in order to produce the action.

    Input:
        state (dict)
            'obs' : actual state representation
            'legal_actions' : possible legal actions to be taken from this state

    Output:
        action (int) : integer representing action id
    '''
    def step(self, state):
        action = self.e_greedy_pick_action(state)
        return self.e_greedy_pick_action(state)


    def reset_hidden(self):
        self.learner_net.init_hidden(1)
        self.target_net.init_hidden(1)


    '''
    Pick an action given a state using epsilon greedy action selection

    Makes call to e_greedy_pick_action to actually select the action

    Input:
        state (dict)
            'obs' : actual state representation
            'legal_actions' : possible legal actions to be taken from this state

    Output:
        action (int) : integer representing action id
    '''
    def e_greedy_pick_action(self, state):
        with torch.no_grad():
            state_obs = torch.FloatTensor(state['obs']).view(1, -1).to(self.device)
            e = self.epsilons[min(self.timestep // self.shift_epsilon_every, len(self.epsilons) - 1)]

            q_values = self.target_net(state_obs).squeeze(1)
            max_action = q_values.argmax(1).item()

            if np.random.uniform() > e or not (max_action in state['legal_actions']):
                probs = remove_illegal(np.ones(self.num_actions), state['legal_actions'])
                action = np.random.choice(self.num_actions, size=None, p=probs)
            else:
                action = max_action

            return action


    '''
    Pick an action given a state. This is to be used during evaluation, so no epsilon greedy.

    Makes call to eval_pick_action to actually select the action

    Input:
        state (dict)
            'obs' : actual state representation
            'legal_actions' : possible legal actions to be taken from this state

    Output:
        action (int) : integer representing action id
        probs (np.array) : softmax distribution over the actions
    '''
    def eval_step(self, state):
        return self.eval_pick_action(state)


    '''
    Pick an action given a state according to max q value.

    Input:
        state (dict)
            'obs' : actual state representation
            'legal_actions' : possible legal actions to be taken from this state

        use_max (bool) : should we return max action or select according to distribution

    Output:
        action (int) : integer representing action id
        probs (np.array) : softmax distribution over the actions
    '''
    def eval_pick_action(self, state, use_max=True):
        with torch.no_grad():
            state_obs = torch.FloatTensor(state['obs']).view(1, -1).to(self.device)

            # Essentially calculate a softmax distribution over the qvalues for legal actions
            exp_qvals = torch.exp(self.target_net(state_obs)[0, 0]).cpu().detach().numpy()
            probs = remove_illegal(exp_qvals, state['legal_actions'])

            if use_max:
                action = np.argmax(probs)
            else:
                action = np.random.choice(self.num_actions, size=None, p=probs)

            return action, probs

    '''
    Add transition to our memory buffer and train the network one batch.

    Input:
        transition (tuple) : tuple representation of a transition --> (state, action, reward, next state, done)

    Output:
        Nothing. Stores transition in buffer, updates network using memory buffer, and updates target network
        depending on what timestep (sequence number) we're at.
    '''
    def add_seq(self, sequence):

        # store sequence of transitions in memory
        if len(sequence) > 0:
            self.memory_buffer.add_seq(sequence)
            self.timestep += 1

            # once we have enough samples, get a sample from our stored memory to train the network
            if self.timestep >= 1000 and self.timestep % 10 == 0:
                batch_loss = self.one_update()
                print('\rstep : {}, loss on batch : {}'.format(self.timestep, batch_loss))

        # copy parameters over every once in awhile
        if self.timestep % self.copy_every == 0:
            self.target_net.load_state_dict(self.learner_net.state_dict())
            self.target_net.eval()
            print('target parameters updated on step : {}'.format(self.timestep))

    '''
    Samples from memory buffer and trains the network one step.

    Input:
        Nothing. Draws sample of sequences from memory buffer to train the network

    Output:
        loss (float) : loss on training batch
    '''
    def one_update(self):
        sequences = self.memory_buffer.sample()

        self.learner_net.train()
        self.optim.zero_grad()

        batch_loss = 0
        for sequence in sequences:


            self.reset_hidden()
            states = [t[0]['obs'] for t in sequence] + [sequence[-1][3]['obs']]
            states = torch.FloatTensor(states).view(len(states), 1, -1).to(self.device)
            actions = torch.LongTensor([t[1] for t in sequence]).view(-1, 1).to(self.device)
            reward = torch.FloatTensor([t[2] for t in sequence]).view(-1).to(self.device)

            # with no gradient, calculate 'true values' for each state in sequence
            # using the target network.
            with torch.no_grad():
                target_q_values = self.target_net(states).detach()
                target_q_values, max_actions = target_q_values.max(-1)
                target_q_values = reward + self.gamma * target_q_values[1:].view(-1)
                target_q_values[-1] = 0.0

            q_values = self.learner_net(states).squeeze(1)[:-1]
            q_values = q_values.gather(-1, actions).view(-1)

            batch_loss += self.criterion(q_values, target_q_values)

        batch_loss.backward()
        self.optim.step()
        self.learner_net.eval()
        return batch_loss.item()


    '''
    Save state dict for networks of DQN agent

    Input:
        file_path (str) : string filepath to save agent at
    '''
    def save_state_dict(self, file_path):
        state_dict = dict()
        state_dict['learner_net'] = self.learner_net.state_dict()
        state_dict['target_net'] = self.target_net.state_dict()

        torch.save(state_dict, file_path)


    '''
    Load agent parameters from filepath

    Input:
        file_path (str) : string filepath to load parameters from
    '''
    def load_from_state_dict(self, filepath):
        state_dict = torch.load(filepath, map_location=self.device)
        self.learner_net.load_state_dict(state_dict['learner_net'])
        self.target_net.load_state_dict(state_dict['target_net'])


'''Sequential memory implementation for recurrent Q Learning network'''
class SeqMemory(object):
    '''Save a series of transitions to use as training examples for the recurrent
    network

    Adapted from memory implementation in:
    https://github.com/datamllab/rlcard/blob/master/rlcard/agents/dqn_agent.py
    '''

    def __init__(self, max_size, batch_size):
        self.max_size = max_size
        self.batch_size = batch_size
        self.memory = []

    def add_seq(self, seq):
        if len(self.memory) == self.max_size:
            self.memory.pop(0)
        self.memory.append(seq)

    def sample(self):
        return sample(self.memory, self.batch_size)
