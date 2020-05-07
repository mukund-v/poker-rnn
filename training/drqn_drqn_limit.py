'''
Script to train implemented DRQN agent against another DRQN agent. Logs in tensorboard.

@author : mukundv
'''

import rlcard
import sys
import torch

from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.utils import tournament
from torch.utils.tensorboard import SummaryWriter

sys.path.append('..')
from agents.drqn_agent import DRQNAgent


# Make environments to train and evaluate models
env = rlcard.make('limit-holdem')
eval_env = rlcard.make('limit-holdem')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize DRQN agents
drqn_agents = []
for i in range(2):
    drqn_agents.append(DRQNAgent(
        num_actions=env.action_num,
        state_shape=env.state_shape,
        hidden_size=128,
        num_layers=2,
        hidden_layers=[256,512],
        lr = .0001,
        batch_size = 32,
        memory_size = 50000,
        copy_every = 300,
        epsilons = [.9, .91, .92, .93, .94, .95, .96, .97, .98, .99, 1],
        shift_epsilon_every = 30000,
        gamma=.97,
        device=device
    ))

# initialize random agent to evaluate against
random_agent = RandomAgent(action_num=eval_env.action_num)
env.set_agents(drqn_agents)
eval_env.set_agents([drqn_agents[0], random_agent])

eval_every = 100
eval_num = 1000
episode_num = 300000

# initialize Tensorboard logger
logger = SummaryWriter('logs/drqn_drqn_agent')

for episode in range(episode_num):

    # reset hidden state of recurrent agents
    for i in range(2):
        drqn_agents[i].reset_hidden()

    # get transitions by playing an episode in env
    trajectories, _ = env.run(is_training=True)

    for i in range(2):
        drqn_agents[i].add_seq(trajectories[i])

    # evaluate against random agent
    if episode % eval_every == 0:
        score = 0
        for i in range(eval_num):
            for j in range(2):
                drqn_agents[j].reset_hidden()
            score += tournament(eval_env, 1)[0]

        logger.add_scalar('reward vs. random agent', score / eval_num, episode)

drqn_agents[0].save_state_dict('./drqn_drqn_agent.pt')
