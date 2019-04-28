import os
import gym
import argparse
import numpy as np
import tensorflow as tf
from her.her import HER
from ddpg.ddpg import DDPG
from stable_baselines.ddpg.policies import MlpPolicy

parser = argparse.ArgumentParser('Deep Coarticulation')
parser.add_argument("-ne", "--no_episodes", type=int, default=10000)
parser.add_argument("-ts", "--timesteps", type=int, default=100)
parser.add_argument("-re", "--render", type=int, default=1)
#parser.add_argument("-ap", "--alpha", type=float, default=0.8)
parser.add_argument("--resume", type=int, default=0)
parser.add_argument("--policy_dir", default="../policies/")
parser.add_argument("--log_dir", default="../logs/")
args = parser.parse_args()

env_names = ['FetchReach-v1', 'FetchPush-v1', 'FetchSlide-v1']
#env = gym.make(args.env_name)
#n_actions = env.action_space.shape[-1]

def train_her(env_name):
    env = gym.make(env_name)
    _ = env.reset()
    print("Training HER on {}".format(env_name))
    model = HER(MlpPolicy, env=env, model_class=DDPG)
    model.learn(total_timesteps=args.timesteps)
    model.save(os.path.join(args.policy_dir, "HER_{}.pol".format(env_name)))
    return model


def test_subpolicy(env_name, her_model):
    env = gym.make(env_name)
    obs = env.reset()
    print("Generating SubPolicies")
    for t in range(args.timesteps):
        action, _states = her_model.predict(obs)
        qvalue = her_model.qvalue(obs, action)
        #print(action, qvalue)
        obs, rewards, done, _ = env.step(action)
        if(done == True):
            obs = env.reset()
        env.render() 

def load_subpolicies(num=(20, 30)):
    subpolicies = []
    print("Loading Subpolicies")
    for env_name in env_names:
        env = gym.make(env_name)
        _ = env.reset()
        path = os.path.join(args.policy_dir, "HER_{}.pol".format(env_name))
        subpl = HER.load(path, env=env)
        subpolicies.append(subpl)
    return subpolicies

def coarticulation(env_name, policy_n, pol_no):
    env = gym.make(env_name)
    _ = env.reset()
    print("Running Coarticulation")
    newpolicy = HER(MlpPolicy, env=env, model_class=DDPG, coarticulation=True)
    
    # NOTE : When coarticulation is set true, HER acts like normal DDPG
    #        and reward is computed as per the coarticulation algo.
    
    newpolicy.learn(total_timesteps=args.timesteps, base_policy=policy_n)
    newpolicy.save(os.path.join(args.policy_dir, "newsubpl_{}.pol".format(pol_no)))
    return newpolicy

def main():
    subpolicies = []
    for env_name in env_names:
        subpolicies.append(train_her(env_name))
        test_subpolicy(env_name, subpolicies[-1])

    newpolicy = coarticulation(env_names[-2], subpolicies[-2], 1)
    '''
    for i in range(len(subpolicies)):
        subpolicies[i] = coarticulation(subpolicies[i], i)
    '''

if __name__ == '__main__':
    main()
