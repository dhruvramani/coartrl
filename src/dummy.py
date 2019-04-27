import os
import gym
import numpy as np
from her.her import HER
from ddpg.ddpg import DDPG
from stable_baselines import SAC
from stable_baselines.common.policies import ActorCriticPolicy


env = gym.make('FetchReach-v1')
model = HER(ActorCriticPolicy, env=env, model_class=SAC)
model.learn(total_timesteps=10000)

model.save("../policies/dummy.pol")
del model

model = HER.load(load_path="../policies/dummy.pol", env=env)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    qvalue = model.qvalue(obs, action)
    print(action, qvalue)
    obs, rewards, done, _ = env.step(action)
    print(done)
    if(done == True):
        obs = env.reset()
    env.render()
