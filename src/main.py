import os
import json
import pickle
import numpy as np
import stable_baslines
import tensorflow as tf
from her.ddpg import DDPG
from her.experiment.train import train

_LOG_PATH = "../logs/"
_ENV_NAME = "FetchReach-v1"
_SPOLICY_PATH = "../policies/"

def load_subpolicies(path=_SPOLICY_PATH, num=(40, 50)):
    