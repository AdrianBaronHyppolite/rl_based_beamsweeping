import math, random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import os
import gym
from gym import logger as gymlogger
from gym.wrappers.record_video import RecordVideo
gymlogger.set_level(40)
from tqdm import tqdm
import matplotlib
import glob
import io
import base64
import warnings
warnings.filterwarnings("ignore") 

#from newenv import RLIAEnv
from newenv import RLIAEnv 
import random

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

env = RLIAEnv() 

from collections import deque

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)                            # Initialization of the buffer of capacity given by the input of the function,' capacity'
    def push(self, state, action, reward, next_state, done):            # Store one single experience (state, action, reward, next_state, done flag) in the buffer
        state      = np.expand_dims(state, 0)                           # If you want to push an experience to the buffer, you have to make sure the 'state' and 'next_state' is 3-dimensional array
        next_state = np.expand_dims(next_state, 0)                      # 'action' and 'reward' are two scalars, the 'done' is 'False' or 'True'
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))    # This method will return you a batch of experiences
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    def __len__(self):                  
        return len(self.buffer)  

replay_buffer = ReplayBuffer(1000)  # Initialize the replay buffer with a capacity of 1000 experiences

class DQN(nn.Module):                                                         # We build our neural network using nn.Module class in pytorch, you may find some examples at https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    def __init__(self, num_actions, num_states):                                          # 'num_action' is the number of all possible actions in the environment. Each action corresponds to one bit output of neural network. If we have 'num_action' actions, the output shape of the neural network should be 'num_action' by 1.
        super(DQN, self).__init__()
        ## student code here
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_states, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_actions))                                       # In this part, you need to define your network layer by layer, here is an great example showing how to implement: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
            
        ## student code end
    def forward(self, x):                                                     # With the network defined, we have to define the forwarding method for the neural network, within which input data will be forwarded in the order of your defined layers, and you need to return final output
        ## student code here
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)
        y = self.layers(x.float())
                                          # define forward procedure of your network
        return y         
    

eval_model = DQN(num_actions=env.action_space.n, num_states=4)
target_model  = DQN(num_actions=env.action_space.n, num_states=4)

if USE_CUDA:
    eval_model = eval_model.cuda()
    target_model  = target_model.cuda()

optimizer = optim.Adam(eval_model.parameters(), lr= 0.001)

def choose_action(state, epsilon):
    nmm =np.random.rand()
    state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True) 
    ## student code here
    if ( nmm< epsilon):                                                         # We can implement exploiting here    
        q_value = eval_model.forward(state)                               # Use .forward() method in DQN class to get the Q values of all actions
        action = q_value.argmax().item()                 # Calculate the index of action with highest Q value. Hint: you can use torch.max()
    else:                                                                                 # We can implement exploring here
        action = np.random.randint(int(env.action_space.n), size=1)[0]   
    return action  

epsilon_initial = 1.0                # The epsilon should be 1 at the beginning        
## student code here
epsilon_by_frame = lambda x: pow((epsilon_initial - 0.001), x)    # Design your function here


def compute_td_loss(batch_size, gamma):
    state, action, reward, next_state, done = replay_buffer.sample(
        batch_size)  # Sample from buffer

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    # Policy network.
    q_policy = eval_model(state)
    q_policy = q_policy.gather(1, action.unsqueeze(1)).squeeze(1)

    # Q values from the target network for the next state
    q_target_next = target_model(next_state)
    q_target_next = q_target_next.max(1)[0].detach()
    q_target = reward + gamma * (1 - done) * q_target_next

    # ## student code here (Note that we have provided some comments based on our implementation, but you do not have to follow them for yours.)
    # # If s_t is non-terminal state (done==Flase), we can follow steps below to calculate the loss:
    # q_values      =                                   # Calculate q values of using eval network with input 'state' using .forward() method
    # next_q_values =                                   # Calculate q values of using eval network with input 'next_state' using .forward() method
    # next_q_state_values =                             # Calculate q values of using target network with input 'next_state' using .forward() method

    # q_value       =         # select the eval q values of actions been played and stored at state s_t, this is the evaluate value
    # next_q_value =  # Use target network to calculate q values of actions that will be selected by eval network at state s_t+1, which is the second Q expression in the loss function above
    # expected_q_value =           # compute the expected value, which is the whole second component in MSE()

    loss_f = nn.MSELoss()
    loss = loss_f(q_policy, q_target)     # Calculate the loss
    # If s_t is terminal state (done==True), we need to calculate the loss using expression mentioned above
    # Calculate the loss
    # student code end
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def update_target(eval_net, target_net):
    global eval_model
    global target_model
    ## student code here
    target_model.load_state_dict(eval_model.state_dict())


kind_experiment = { 
    0: "Random",
    1: "Exhaustive search",
    2: "DDQN",
}

chosen_strategy = { 
    0: "9x9",
    1: "25x25",
    2: "7x7",
    3: "Exhaustive search",
}

def appendToDf(df, type, ts_begin, ts_end, spectral_efficiency, episode, reward, strategy, distance, cbsp, angle):
    df.loc[len(df)] = [type, kind_experiment[type],ts_begin, ts_end, ts_end-ts_begin, spectral_efficiency, episode, reward, strategy, chosen_strategy[strategy], distance, cbsp, angle]



measurements = pd.DataFrame(columns=["type",  "type_str",
                  "ts_begin", "ts_end", "ts_diff", "spectral_efficiency", "episode", "reward", "strategy", "strategy_str", "distance", "CBSP", "angle"])


replay_buffer = ReplayBuffer(1000)  # clear the buffer
# student code here
num_frames = 10000                        # Suggested range: 2000 - 10000
batch_size = 64                           # Suggested range: 8 - 128
GAMMA = 0.9                      # Suggested range: >0.9
NETWORK_UPDATE_PERIOD = 50        # Suggested range: 50 - 300
# student code end

episodes = 3000 
losses = []
all_rewards = []
episode_reward = 0

beam9stren = []
beam25stren = []
beamexhaustivestren = []

actions = ["9beam", "25beam", "exhaustive"]

print("Starting training")
for i in range(episodes):
    state = env.reset()
    env.ptx_dbm = 10
    for i in range(10):
        epsilon = .1
        #epsilon = epsilon_by_frame(i)
        action = choose_action(state, epsilon)                  # pick action
        next_state, reward, done, _ = env.teststep(
            action)          # interact with environment
        replay_buffer.push(state, action, reward, next_state,
                        done)  # store experience 
        state = next_state
        episode_reward += reward
        if len(replay_buffer) > batch_size:                             # start training
            loss = compute_td_loss(batch_size, GAMMA)
            losses.append(loss.cpu().data.numpy())
        # update target network using eval network
        if i % NETWORK_UPDATE_PERIOD == 0:
            update_target(eval_model, target_model)



    actions = ["9beam", "25beam","7by7","exhaustive"]

# testing
print("Starting testing")
epinumex = 0
type_measurement = 1
for _ in range(2000):
        ts_begin = time.process_time()
        action = 2
        observation, reward, _, _ = env.teststep(action)
        ts_end = time.process_time() 
        epinumex = epinumex+1
        if env.ue_loc[1] != -1:
            appendToDf(df=measurements, type=type_measurement,ts_begin=ts_begin, ts_end=ts_end, spectral_efficiency=observation[0], episode = epinumex, reward=reward, strategy=int(action), distance=env.state[3], cbsp = env.cbsp, angle=env.state[2])




epinum = 0
type_measurement = 2
for i in range(2000):
    ts_begin = time.process_time()
    q_value = target_model.forward(state)                               # Use .forward() method in DQN class to get the Q values of all actions
    action = q_value.argmax().item()      
    # action = choose_action(state, 1)
    state, reward, done, _ = env.teststep(action)
    ts_end = time.process_time()
    epinum = epinum+1
    #only save measurements where env.ue_loc[1] does not equal -1
    if env.ue_loc[1] != -1:
        appendToDf(df=measurements, type=type_measurement,ts_begin=ts_begin, ts_end=ts_end, spectral_efficiency=state[0], episode = epinum, reward=reward, strategy=int(action), distance=state[3], cbsp = env.cbsp, angle=env.state[2])
    
env.close()

print("Saving measurements")
measurements.to_csv("measurements.csv")

grouped = measurements.groupby(['type', 'strategy', 'strategy_str']).sum()

exhaustive = measurements.query("type == 1")
ddqn = measurements.query("type == 2")
g = sns.lineplot(data=exhaustive,x="angle", y="CBSP", label="Exhaustive search")
g = sns.lineplot(data=ddqn,x="angle", y="CBSP", label="DDQN")
g.set(ylabel="CBSP (Bits/s/Hz)", xlabel="angle", title="CBSP as UE moves radially")
#plt.show(g)

g.figure.savefig("CBSP.png")

print("Done!")