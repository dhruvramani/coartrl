import os
import gym
import numpy as np
from her.her import HER
from ddpg.ddpg import DDPG
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv


env = gym.make('FetchReach-v1')
env = DummyVecEnv([lambda: env])
model = HER(MlpPolicy, env=env, model_class=DDPG)
model.learn(total_timesteps=100)

model.save("../policies/dummy.pol")
del model

model = DDPG.load("../policies/dummy.pol")

obs = env.reset()
for i in range(10):
    action, _states = model.predict(obs)
    obs, rewards, done, _ = env.step(action)
    env.render()
