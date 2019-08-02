import gym
import numpy as np

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC

#env = gym.make('Pendulum-v0')

def stable_sac(env):
	env = DummyVecEnv([lambda: env])
	model = SAC(MlpPolicy, env, verbose=1)
	model.learn(total_timesteps=50000, log_interval=10)
	model.save("sac_jaco")
