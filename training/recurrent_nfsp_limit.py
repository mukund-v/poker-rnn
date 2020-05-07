'''
Script to train implemented Recurrent NFSP agent. Logs in tensorboard.

@author : mukundv
'''

import rlcard
import sys
import torch

from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.utils import set_global_seed, tournament
from torch.utils.tensorboard import SummaryWriter

sys.path.append('..')
from agents.recurrent_nfsp_agent import RNFSPAgent


# Make environments to train and evaluate models
env = rlcard.make('limit-holdem')
eval_env = rlcard.make('limit-holdem')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize RNFSP agents
nfsp_agents = []
for i in range(2):
    nfsp_agents.append(RNFSPAgent(
        num_actions=env.action_num,
        state_shape=env.state_shape,
        recurrent_layers=1,
        hidden_size=1024,
        ap_hidden_layers=[1024, 512],
        av_hidden_layers=[1024, 512],
        ap_lr = .005,
        av_lr = .06,
        batch_size = 64,
        rl_memory_size = 40000,
        sl_memory_size = 500000,
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
logger = SummaryWriter('logs/rnfsp_agent1')


for episode in range(episode_num):

    # set policy regime for NFSP agent and reset hidden state
    for agent in nfsp_agents:
        agent.set_policy()
        agent.reset_hidden()

    # get transitions by playing an episode in env
    trajectories, _ = env.run(is_training=True)

    for i in range(2):
        nfsp_agents[i].add_seq(trajectories[i])

    # evaluate against the random agent
    if episode % eval_every == 0:
        nfsp_agents[0].set_policy('average_policy')
        results = 0
        for i in range(eval_num):
            nfsp_agents[0].reset_hidden()
            results += tournament(eval_env, 1)[0]
        logger.add_scalar('reward vs. random agent', results / eval_num, episode)

    if episode % 50000 == 0:
        nfsp_agents[0].save_state_dict('./rnfsp_agent1.pt')
