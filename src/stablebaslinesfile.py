import gym
from newenv import RLIAEnv
from stable_baselines3 import DQN
import tensorboard as tb
import os


env = RLIAEnv() 

model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=4000)
TIMESTEPS = 10000


episodes = 10
prestate = env.sweep(env.codebook.keys())
env.preid = prestate[1]
for i in range(1):
      obs = env.reset()
      prestate = env.sweep(env.codebook.keys())
      env.preid = prestate[1]
      for ep in range(episodes):
            done = False
            while not done:                                                                                                     
                obs, reward, done, info = env.step((env.action_space.sample())) # take a random action
                env.preid = env.state[1]

env.close()


# obs, reward, done, info = env.step((env.action_space.sample()), 31, ([15,0])) # take a random action
#     print(reward)