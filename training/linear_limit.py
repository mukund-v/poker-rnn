'''
Script to train implemented DQN agent against another DQN agent. Logs in tensorboard.

@author : mukundv
'''

import rlcard
import sys
import torch

from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.utils import tournament
from torch.utils.tensorboard import SummaryWriter

sys.path.append('..')
from agents.linear_q_agent import LinearQAgent


# Make environments to train and evaluate models
env = rlcard.make('limit-holdem')
eval_env = rlcard.make('limit-holdem')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize linear agents
linear_agents = []
for i in range(2):
    linear_agents.append(LinearQAgent(
        num_actions=env.action_num,
        state_shape=env.state_shape,
        lr = .0001,
        batch_size = 64,
        memory_size = 100000,
        copy_every = 300,
        epsilons = [.9, .91, .92, .93, .94, .95, .96, .97, .98, .99, 1],
        shift_epsilon_every = 30000,
        gamma=.97,
        device=device
    ))

# initialize random agent to evaluate against
random_agent = RandomAgent(action_num=eval_env.action_num)
env.set_agents(linear_agents)
eval_env.set_agents([linear_agents[0], random_agent])

eval_every = 100
eval_num = 1000
episode_num = 300000

# initialize Tensorboard logger
logger = SummaryWriter('logs/linear_linear_agent')


for episode in range(episode_num):

    # get transitions by playing an episode in env
    trajectories, _ = env.run(is_training=True)

    for i in range(2):
        for trajectory in trajectories[i]:
            linear_agents[i].add_transition(trajectory)

    # evaluate against random agent
    if episode % eval_every == 0:
        result = tournament(eval_env, eval_num)[0]
        logger.add_scalar('reward vs. random agent', result, episode)

linear_agents[0].save_state_dict('./linear_linear_agent.pt')
