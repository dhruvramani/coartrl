import os
import gym
import numpy as np
import tensorflow as tf
from her.her import HER
from ddpg.ddpg import DDPG
from stable_baselines.ddpg.policies import MlpPolicy

_LOG_PATH = "../logs/"
_ENV_NAME = "FetchReach-v1"
_SPOLICY_PATH = "../policies/"
_EPISODE_LEN = 10000

env = gym.make(_ENV_NAME)
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

def train_her(save_path):

def load_subpolicies(path=_SPOLICY_PATH, num=(40, 50)):

def coarticulation(policy_n, policy_n1):
    observ = env.reset()
    newpolicy = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)

    for e in range(_EPISODE_LEN):
        action, _states = model.predict(observ)
        qvalue = model.qvalue(obs, action)
        print(action, qvalue)
        observ, rewards, done, _ = env.step(action)