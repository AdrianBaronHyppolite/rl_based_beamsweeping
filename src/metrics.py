# %%
import seaborn as sns 
import pandas as pd 
import numpy as np
import time

sns.color_palette('colorblind')
penguins = sns.load_dataset("penguins")

# %%
print(penguins.head)

# %%
sns.histplot(data=penguins, x="flipper_length_mm", hue="species", multiple="stack")

# %%
from mab import eps_bandit
from environment import RLIAEnv 
from contextualMab import contextual_bandit

env = RLIAEnv() 

kind_experiment = { 
    0: "Random",
    1: "MAB",
    2: "Contextual_MAB",
}

def appendToDf(df, type, ts_begin, ts_end, spectral_efficiency, reward):
    df.loc[len(df)] = [type, kind_experiment[type],ts_begin, ts_end, ts_end-ts_begin, spectral_efficiency, reward]

measurements = pd.DataFrame(columns=["type",  "type_str",
                  "ts_begin", "ts_end", "ts_diff", "spectral_efficiency", "reward"]) 

episodes = 1000
training_episodes = 1000 



# %%
# Training MAB
eps=0.1
mab_agent = eps_bandit(k=4, eps=eps, iters=training_episodes) 

for i in range(training_episodes): 
    action = mab_agent.act(eps)
    _ , r, _ , _ = env.step(action=action)

    # Collect reward.
    mab_agent.learn(reward=r, action=action) 

# %%
# Training Contextual MAB    
eps=0.1
lr= 0.15
contextual_agent = contextual_bandit(k=4, eps=eps, iters=training_episodes, hidden_size=32, 
    state_size=3, learning_rate=lr)  

for i in range(training_episodes): 
    flat_state = np.array([env.state[0][0][0],env.state[1][0][0],env.state[2]])
    action = contextual_agent.act(flat_state)
    _ , r, _ , _ = env.step(action=action)

    # Collect reward.
    contextual_agent.learn(reward=r, state=flat_state, action=action) 

# %%
# random measurement
type_measurement = 0
for episode in range(episodes):
    ts_begin = time.process_time()
    action = env.action_space.sample() 
    observation, reward, _, _ = env.step(action)
    ts_end = time.process_time() 

    appendToDf(df=measurements, type=type_measurement,ts_begin=ts_begin, ts_end=ts_end, spectral_efficiency=observation[2], reward=reward)

# MAB measurement
type_measurement = 1
for _ in range(episodes):
    # Estimate channel.
    ts_begin = time.process_time()
    action = mab_agent.act()
    observation, reward, _, _ = env.step(action=action)
    appendToDf(df=measurements, type=type_measurement,ts_begin=ts_begin, ts_end=ts_end, spectral_efficiency=observation[2], reward=reward)

# MAB measurement
type_measurement = 2
for _ in range(episodes):
    # Estimate channel.
    ts_begin = time.process_time()
    flat_state = np.array([env.state[0][0][0],env.state[1][0][0],env.state[2]])
    action = contextual_agent.act(flat_state)
    observation, reward, _, _ = env.step(action=action)
    appendToDf(df=measurements, type=type_measurement,ts_begin=ts_begin, ts_end=ts_end, spectral_efficiency=observation[2], reward=reward)

# %%
print(measurements.head)

# %%
sns.displot(data=measurements, x="ts_diff", hue="type_str", col="type_str")

# %%
sns.displot(data=measurements, x="spectral_efficiency", hue="type_str", col="type_str")


