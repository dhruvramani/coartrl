import os
import gym
import numpy as np
from her.her import HER
from ddpg.ddpg import DDPG
from stable_baselines.ddpg.policies import MlpPolicy


env = gym.make('FetchReach-v1')
model = HER(MlpPolicy, env=env, model_class=DDPG)
model.learn(total_timesteps=10000)

#model.save("../policies/dummy.pol")
#del model

#model = DDPG.load("../policies/dummy.pol")

obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs)
    qvalue = model.qvalue(obs, action)
    print(action, qvalue)
    obs, rewards, done, _ = env.step(action)
    print(done)
    if(done == True):
        obs = env.reset()
    env.render()
