import os
import gym
import numpy as np
from her.her import HER
from stable_baselines import DDPG
from stable_baselines.common.policies import DDPGPolicy

env = gym.make('FetchReach-v1')
model = HER(DDPGPolicy, env=env, model_class=DDPG)
mdoel.learn(total_timesteps=5000)

obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, done, _ = env.step(action)
    env.render()
