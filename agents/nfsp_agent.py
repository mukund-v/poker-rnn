'''
PyTorch implementation of Neural Fictitious Self Play agent for rlcard platform

The action-value network is approximated with essentially a Q-Learning agent.
The average-policy network is appoximated with a fully connected network.

@author : mukundv

Referenced and used util functions from:

https://github.com/datamllab/rlcard/blob/master/rlcard/agents/dqn_agent.py
https://github.com/datamllab/rlcard/blob/master/rlcard/utils/utils.py
'''

import numpy as np
import torch

from .models.dqnet import DQNet
from rlcard.agents.dqn_agent import Memory
from rlcard.utils.utils import remove_illegal
from random import sample
from torch import nn


'''
NFSP agent. uses agent wrapper as specified:
http://rlcard.org/development.html#developping-algorithms

Parameters:

    num_actions (int) : how many possible actions

    state_shape (list) : tensor shape of state

    ap_hidden_layers (list) : hidden layer sizes to use for average policy net

    av_hidden_layers (list) : hidden layer sizes to use for average value net

    ap_lr (float) : learning rate to use for training average policy net

    av_lr (float) : learning rate to use for training action value net

    batch_size (int) : batch sizes to use when training networks

    rl_memory_size (int) : max number of experiences to store in reinforcement learning memory buffer

    sl_memory_size (int) : max number of experiences to store in supervised learning memory buffer

    copy_every (int) : how often to copy parameters to target network

    epsilons (list) : list of epsilon values to use over training period

    shift_epsilon_every (int) : how often should we shift our epsilon value

    eta (float) : anticipatory parameter for NFSP

    gamma (float) : discount parameter

    device (torch.device) : device to put models on
'''
class NFSPAgent():
    def __init__(self,
                 num_actions,
                 state_shape,
                 ap_hidden_layers,
                 av_hidden_layers,
                 ap_lr = .001,
                 av_lr = .0001,
                 batch_size = 64,
                 rl_memory_size = 30000,
                 sl_memory_size = 1000000,
                 copy_every = 1000,
                 epsilons = [.92, .93, .94, .95, .96, .97, .98, .99],
                 shift_epsilon_every = 40000,
                 eta = .1,
                 gamma=.9,
                 device=None):

        self.num_actions = num_actions
        self.batch_size = batch_size
        self.copy_every = copy_every
        self.epsilons = epsilons
        self.shift_epsilon_every = shift_epsilon_every
        self.eta = eta
        self.gamma = gamma
        self.device = device
        self.use_raw = False

        # average policy can be modeled as a Deep Q Network and we take softmax after final layer
        self.average_policy = DQNet(state_shape, num_actions, ap_hidden_layers, activation='relu').to(self.device)
        self.average_policy.eval()

        # action value and target network are Deep Q Networks
        self.action_value = DQNet(state_shape, num_actions, av_hidden_layers, activation='relu').to(self.device)
        self.target_net = DQNet(state_shape, num_actions, av_hidden_layers, activation='relu').to(self.device)
        self.target_net.eval()

        # initialize loss functions
        self.ap_criterion = nn.CrossEntropyLoss()
        self.av_criterion = nn.MSELoss()

        # initialize optimizers
        self.ap_optim = torch.optim.Adam(self.average_policy.parameters(), lr=ap_lr)
        self.av_optim = torch.optim.Adam(self.action_value.parameters(), lr=av_lr)

        # initialize memory buffers
        self.rl_buffer = Memory(rl_memory_size, batch_size)
        self.sl_buffer = ReservoirMemoryBuffer(sl_memory_size, batch_size, .25)

        # current policy
        self.policy = None

        self.softmax = torch.nn.Softmax(dim=1)

        self.timestep = 0


    '''Set policy parameter

    Input :
        policy (str) : policy to use. sets according to anticipatory parameter on default.

    Output :
        None, sets policy parameter
    '''
    def set_policy(self, policy=None):
        # set policy according to string
        if policy and policy in ['average_policy', 'best_response', 'greedy_average_policy']:
            self.policy = policy
        else:
            self.policy = 'best_response' if np.random.uniform() <= self.eta else 'average_policy'


    '''
    Given state, produce actions to generate training data. Choose action according to set policy parameter.

    Input:
        state (dict)
            'obs' : actual state representation
            'legal_actions' : possible legal actions to be taken from this state

    Output:
        action (int) : integer representing action id
    '''
    def step(self, state):
        if self.policy == 'average_policy':
            return self.ap_pick_action(state)[0]
        elif self.policy == 'best_response':
            return self.e_greedy_pick_action(state)

    '''
    Pick an action given a state using epsilon greedy action selection using target network

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

            q_values = self.target_net(state_obs)
            max_action = q_values.argmax(1).item()

            if np.random.uniform() > e or not (max_action in state['legal_actions']):
                probs = remove_illegal(np.ones(self.num_actions), state['legal_actions'])
                action = np.random.choice(self.num_actions, size=None, p=probs)
            else:
                action = max_action
            return action


    '''
    Pick an action given a state using the average policy network

    Input:
        state (dict)
            'obs' : actual state representation
            'legal_actions' : possible legal actions to be taken from this state

    Output:
        action (int) : integer representing action id
    '''
    def ap_pick_action(self, state):
        with torch.no_grad():
            state_obs = torch.FloatTensor(state['obs']).view(1, -1).to(self.device)
            softmax_qvals = self.softmax(self.average_policy(state_obs))[0].cpu().detach().numpy()
            probs = remove_illegal(softmax_qvals, state['legal_actions'])
            action = np.random.choice(self.num_actions, size=None, p=probs)
            return action, probs


    '''
    Pick an action greedily given a state using the average policy network

    Input:
        state (dict)
            'obs' : actual state representation
            'legal_actions' : possible legal actions to be taken from this state

    Output:
        action (int) : integer representing action id
    '''
    def greedy_ap_pick_action(self, state):
        with torch.no_grad():
            state_obs = torch.FloatTensor(state['obs']).view(1, -1).to(self.device)
            softmax_qvals = self.softmax(self.average_policy(state_obs))[0].cpu().detach().numpy()
            probs = remove_illegal(softmax_qvals, state['legal_actions'])
            action = np.argmax(probs)
            return action, probs


    '''
    Pick an action given a state according to set policy. This is to be used during evaluation, so no epsilon greedy.

    Makes call to eval_pick_action or average_policy to actually select the action

    Input:
        state (dict)
            'obs' : actual state representation
            'legal_actions' : possible legal actions to be taken from this state

    Output:
        action (int) : integer representing action id
        probs (np.array) : softmax distribution over the actions
    '''
    def eval_step(self, state):
        if self.policy == 'average_policy':
            return self.ap_pick_action(state)
        elif self.policy == 'best_response':
            return self.eval_pick_action(state)
        elif self.policy == 'greedy_average_policy':
            return self.greedy_ap_pick_action(state)


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
            softmax_qvals = self.softmax(self.target_net(state_obs))[0].cpu().detach().numpy()
            probs = remove_illegal(softmax_qvals, state['legal_actions'])

            if use_max:
                action = np.argmax(probs)
            else:
                action = np.random.choice(self.num_actions, size=None, p=probs)

            return action, probs


    '''
    Add transition to our memory buffers and train the networks one batch.

    Input:
        transition (tuple) : tuple representation of a transition --> (state, action, reward, next state, done)

    Output:
        Nothing. Stores transition in the buffers, updates networks using memory buffers, and updates target network
        depending on what timestep we're at.
    '''
    def add_transition(self, transition):
        state, action, reward, next_state, done = transition

        if self.policy == 'best_response':
            self.sl_buffer.add_sa(state['obs'], action)
        self.rl_buffer.save(state['obs'], action, reward, next_state['obs'], done)
        self.timestep += 1

        if len(self.rl_buffer.memory) >= 1000 and self.timestep % 64 == 0:
            av_loss = self.action_value_update()
            print('step : {} action value parameters updated'.format(self.timestep))
        if len(self.sl_buffer.memory) >= 1000 and self.timestep % 64 == 0:
            ap_loss = self.average_policy_update()
            print('step : {} average policy updated'.format(self.timestep))

        if self.timestep % self.copy_every == 0:
            print('target net params updated')
            self.target_net.load_state_dict(self.action_value.state_dict())
            self.target_net.eval()

    '''
    Samples from reinforcement learning memory buffer and trains the action value network one step.

    Input:
        Nothing. Draws sample from rl buffer to train the network

    Output:
        loss (float) : loss on training batch
    '''
    def action_value_update(self):
        states, actions, rewards, next_states, dones = self.rl_buffer.sample()

        states = torch.FloatTensor(states).view(self.batch_size, -1).to(self.device)
        next_states = torch.FloatTensor(next_states).view(self.batch_size, -1).to(self.device)
        dones = (1 - torch.FloatTensor(dones)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)

        with torch.no_grad():
            target_net_qvals = self.target_net(next_states)
            next_state_val, actions = target_net_qvals.max(1)
            true_vals = rewards + self.gamma * dones * next_state_val
            true_vals = true_vals.detach()

        self.action_value.train()
        self.av_optim.zero_grad()

        outputs = self.action_value(states).gather(1, actions.view(-1, 1)).view(-1)

        loss = self.av_criterion(outputs, true_vals)
        loss.backward()
        self.av_optim.step()

        self.action_value.eval()
        return loss.item()


    '''
    Samples from supervised learning memory buffer and trains the average policy network one step.

    Input:
        Nothing. Draws sample from sl buffer to train the network

    Output:
        loss (float) : loss on training batch
    '''
    def average_policy_update(self):
        sample = self.sl_buffer.sample()

        states = [s[0] for s in sample]
        actions = [s[1] for s in sample]

        states = torch.FloatTensor(states).view(self.batch_size, -1).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)

        self.average_policy.train()
        self.ap_optim.zero_grad()

        outputs = self.average_policy(states)
        loss = self.ap_criterion(outputs, actions)

        loss.backward()
        self.ap_optim.step()

        self.average_policy.eval()

        return loss.item()


    '''
    Save state dict for networks of NFSP agent

    Input:
        file_path (str) : string filepath to save agent at
    '''
    def save_state_dict(self, file_path):
        state_dict = dict()
        state_dict['action_value'] = self.action_value.state_dict()
        state_dict['average_policy'] = self.average_policy.state_dict()
        state_dict['target_net'] = self.target_net.state_dict()

        torch.save(state_dict, file_path)


    '''
    Load agent parameters from filepath

    Input:
        file_path (str) : string filepath to load parameters from
    '''
    def load_from_state_dict(self, filepath):
        state_dict = torch.load(filepath, map_location=self.device)
        self.action_value.load_state_dict(state_dict['action_value'])
        self.average_policy.load_state_dict(state_dict['average_policy'])
        self.target_net.load_state_dict(state_dict['target_net'])


'''Reservoir sampling implementation with exponential bias toward newer examples.'''
class ReservoirMemoryBuffer():
    '''Save a series of state action pairs to use in training of average policy network'''
    def __init__(self, max_size, batch_size, rep_prob):
        self.max_size = max_size
        self.batch_size = batch_size
        self.rep_prob = rep_prob
        self.memory = []

    def add_sa(self, state, action):
        if len(self.memory) < self.max_size:
            self.memory.append((state, action))
        elif np.random.uniform() <= self.rep_prob:
            i = int(np.random.uniform() * self.max_size)
            self.memory[i] = (state, action)

    def sample(self):
        return sample(self.memory, self.batch_size)
