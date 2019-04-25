import os
import gym
import numpy as np
from her.her import HER
from ddpg.ddpg import DDPG
from stable_baselines.ddpg.policies import MlpPolicy

env = gym.make('FetchReach-v1')
model = HER(MlpPolicy, env=env, model_class=DDPG)
model.learn(total_timesteps=50000)

obs = env.reset()
for i in range(10000):
    action, _states = model.predict(obs)
    obs, rewards, done, _ = env.step(action)
    env.render()
