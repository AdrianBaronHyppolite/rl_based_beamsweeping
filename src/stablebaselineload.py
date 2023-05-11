import gym
from stable_baselines3 import DQN
from newenv import RLIAEnv

env = RLIAEnv() 
env.reset()

models_dir = "models/DQN"

model_path = f"src/models/DQN/10000.zip"

model = DQN.load(model_path, env=env)

#model = DQN('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=100000)



episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        #print(obs)
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action) # take a random action
        env.preid = env.state[1]
        print("signal strength: ", env.state[0])
        print("action choice: ", action)
        print("ue location", env.ue_loc)
        print("reward: ", reward)
        print("                   ")


env.close()