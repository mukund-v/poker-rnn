# POKER-RNN

Thank you to the developers of [RLCard](https://github.com/datamllab/rlcard) for a great platform to develop agents and for the great examples!

The file structure is layed out in the following manner

```
+ root
|--- + agents - the various reinforcement learning agents we wrote over the course of the project
|    |--- + models - the PyTorch models used in the agents
|    |    |--- dqnet.py - PyTorch implementation of Deep Q network used in DQNAgent, NFSP agent
|    |    |--- drqnet.py - PyTorch implementation of Deep Recurrent Q network used in DRQNAgent, RNFSP agent
|    |    |--- lrnet.py - PyTorch implementation of Linear Q Network used in LinearQAgent
|    |--- dqn_agent.py - implementation of DQN agent for RLCard
|    |--- drqn_agent.py - implementation of DRQN agent for RLCard
|    |--- linear_q_agent.py - implementation of Linear Q agent for RLCard
|    |--- nfsp_agent.py - implementation of NFSP agent for RLCard
|    |--- recurrent_nfsp_agent.py - implementation of recurrent NFSP agent for RLCard
|--- + training - scripts to train the various models
|    |--- dqn_dqn_limit.py - train a DQN agent against another DQN agent
|    |--- drqn_drqn_limit.py - train a DRQN agent against another DRQN agent
|    |--- linear_limit.py - train a LQA agent against another LQA agent
|    |--- nfsp_limit.py - train a NFSP agent against another NFSP agent
|    |--- recurrent_nfsp_limit.py - train a RNFSP agent against another RNFSP agent
|--- reqs.txt - file to pip install dependencies for the project
```

## Running our agents
In order to run our agents, simply create a new python environment using virtual env. You can download it and create a new environment named 'env' using the following commands:

```
python3 -m pip install --user virtualenv
python3 -m venv env
```

Finally, activate the virtual environment using:

```
source env/bin/activate
```


Next, pip install the required packages in the reqs.txt file. This requires all necessary packages for our project. We can do this using the command:

```
pip install -r reqs.txt
```

Finally, to run any of our agents, simply cd to the training directory and run the corresponding script. For example, DQN:

```
python dqn_dqn_limit.py
```

and the model will start training. Reward against a random agent will also be plotted in Tensorboard in the training/logs directory.
