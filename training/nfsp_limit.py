'''
Script to train implemented NFSP agent. Logs in tensorboard.

@author : mukundv
'''

import rlcard
import sys
import torch

from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.utils import tournament
from torch.utils.tensorboard import SummaryWriter

sys.path.append('..')
from agents.nfsp_agent import NFSPAgent


# Make environments to train and evaluate models
env = rlcard.make('limit-holdem')
eval_env = rlcard.make('limit-holdem')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize NFSP agents
nfsp_agents = []
for i in range(2):
    nfsp_agents.append(NFSPAgent(
        num_actions=env.action_num,
        state_shape=env.state_shape,
        ap_hidden_layers=[1024, 512, 512],
        av_hidden_layers=[1024, 512, 512],
        ap_lr = .01,
        av_lr = .1,
        batch_size = 256,
        rl_memory_size = 30000,
        sl_memory_size = 1000000,
        copy_every = 1000,
        epsilons = [.92, .93, .94, .95, .96, .97, .98, .99],
        shift_epsilon_every = 40000,
        eta = .2,
        gamma=.99,
        device=device
    ))

# initialize random agent to evaluate against
random_agent = RandomAgent(action_num=eval_env.action_num)
env.set_agents(nfsp_agents)
eval_env.set_agents([nfsp_agents[0], random_agent])

eval_every = 100
eval_num = 1000
episode_num = 500000

# initialize Tensorboard logger
logger = SummaryWriter('logs/nfsp_agent2')

for episode in range(episode_num):

    # set policy regime for NFSP agents
    for agent in nfsp_agents:
        agent.set_policy()

    # get transitions by playing an episode in env
    trajectories, _ = env.run(is_training=True)

    for i in range(2):
        for trajectory in trajectories[i]:
            nfsp_agents[i].add_transition(trajectory)

    # evaluate against random agent with average policy
    if episode % eval_every == 0:
        nfsp_agents[0].set_policy('average_policy')
        result = tournament(eval_env, eval_num)[0]
        logger.add_scalar('reward vs. random agent', result, episode)

nfsp_agents[0].save_state_dict('./nfsp_agent2.pt')
