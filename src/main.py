import os
import gym
import numpy as np
from her.her import HER
from ddpg.ddpg import DDPG
from stable_baselines.ddpg.policies import MlpPolicy

_LOG_PATH = "../logs/"
_ENV_NAME = "FetchReach-v1"
_SPOLICY_PATH = "../policies/"

def load_subpolicies(path=_SPOLICY_PATH, num=(40, 50)):
    