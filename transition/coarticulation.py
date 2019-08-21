import sys
import os
import os.path as osp
import tensorflow as tf

import baselines.common.tf_util as U

from primitive_policy import PrimitivePolicy
from util import *
import rollouts

from sac.sac import sac
from sac.sac_original import sac_original
from sac.utils.run_utils import setup_logger_kwargs

from stable_sac import stable_sac

def coarticulation_trpo(env, primitive_pi, config, prim_props):
    ob = env.reset()
    primitive_env_name = primitive_pi.ob_env_name
    coart_pi = PrimitivePolicy(name="%s/coartpi" % primitive_env_name, env=env, ob_env_name=primitive_env_name, config=config)
    coart_oldpi = PrimitivePolicy(name="%s/coart_oldpi" % primitive_env_name, env=env, ob_env_name=primitive_env_name, config=config)

    # BIG TIME HACK - to avoid debugging
    config.is_train = True

    var_list = prim_props[0] + coart_pi.get_variables() + coart_oldpi.get_variables()
    coart_path = osp.expanduser(osp.join(config.coart_dir, config.coart_name))
    
    from trainer_coart import RLTrainer

    trainer = RLTrainer(env, coart_pi, coart_oldpi, primitive_pi, config)
    rollout = rollouts.traj_segment_generator_coart(env, primitive_pi, coart_pi, alpha=config.coart_alpha, stochastic=not config.is_collect_state, config=config)

    if(not config.coart_start):
        ckpt_path = load_model(coart_path, var_list)

    print("Testing Primitive")
    if(prim_props[3] == None):
        prim_props[3] = ckpt_path
    prim_props[2].evaluate(prim_props[1], ckpt_num=prim_props[3].split('/')[-1])

    if(config.coart_train):
        print("Training Co-Articulations")
        trainer.train(rollout)

    print("Testing Coart")
    trainer.evaluate(rollout, ckpt_num=ckpt_path.split('/')[-1])

def coarticulation_sac(env, primitive_pi, config):
    ob = env.reset()
    test_env = make_env(config.env, config)
    logger_kwargs = setup_logger_kwargs(config.sac_exp_name, 0)
    ac_kwargs = dict(hidden_sizes=[config.sac_hid] * config.sac_l)

    print("Training Co-Articulations")
    sac(env, test_env=test_env, primitive_pi=primitive_pi, ac_kwargs=ac_kwargs, alpha=0.0, logger_kwargs=logger_kwargs)

def run_sac_original(env, config):
    test_env = make_env(config.env, config)
    logger_kwargs = setup_logger_kwargs(config.sac_exp_name, 0)
    ac_kwargs = dict(hidden_sizes=[config.sac_hid] * config.sac_l)
    sac_original(env, test_env=test_env, ac_kwargs=ac_kwargs, alpha=0.0, logger_kwargs=logger_kwargs, render=config.render)

def run_stable_sac(env, config):
    ob = env.reset()
    stable_sac(env)

def coarticulation_new(env, config):
    ob = env.reset()

    p1 = PrimitivePolicy(name="%s/pi" % config.primitive_envs[0], env=env, ob_env_name=config.primitive_envs[0], config=config)
    p1_old = PrimitivePolicy(name="%s/oldpi" % config.primitive_envs[0], env=env, ob_env_name=config.primitive_envs[0], config=config)
    p2 = PrimitivePolicy(name="%s/pi" % config.primitive_envs[1], env=env, ob_env_name=config.primitive_envs[1], config=config)
    p2_old = PrimitivePolicy(name="%s/oldpi" % config.primitive_envs[1], env=env, ob_env_name=config.primitive_envs[1], config=config)

    p1_vars, p2_vars = p1.get_variables() + p1_old.get_variables(), p2.get_variables() + p2_old.get_variables()
    p1_path = osp.expanduser(osp.join(config.primitive_dir, config.primitive_paths[0]))
    p2_path = osp.expanduser(osp.join(config.primitive_dir, config.primitive_paths[1]))

    # NOTE : CHANGE THIS TO SAC
    #coart_pi = PrimitivePolicy(name="%s/coartpi" % config.primitive_envs[1], env=env, ob_env_name=config.primitive_envs[1], config=config)
    #coart_oldpi = PrimitivePolicy(name="%s/coart_oldpi" % config.primitive_envs[1], env=env, ob_env_name=config.primitive_envs[1], config=config)

    #var_list = coart_pi.get_variables() + coart_oldpi.get_variables()
    #coart_path = osp.expanduser(osp.join(config.coart_dir, config.coart_name))

    from trainer_rl import RLTrainer

    initial_rollouts = config.num_rollouts
    
    #config.num_rollouts = 20 # NOTE : important, to execute primitive once
    # trainer_p1 = RLTrainer(env, p1, p1_old, config)
    # rollout_p1 = rollouts.traj_segment_generator_rl(env, p1, stochastic=not config.is_collect_state, config=config)

    # #if(not config.coart_start):
    # #    coart_path = load_model(coart_path, var_list)
    
    # p1c_path = load_model(p1_path, p1_vars)
    # trainer_p1.evaluate(rollout_p1, ckpt_num=p1c_path.split('/')[-1])
    
    # config.num_rollouts = initial_rollouts # NOTE : set it back

    # # NOTE : Run SAC here
    # #run_sac_original(env, config)

    # config.num_rollouts = 20 # NOTE : important, to execute second primitive once
    # trainer_p2 = RLTrainer(env, p2, p2_old, config)
    # rollout_p2 = rollouts.traj_segment_generator_rl(env, p2, stochastic=not config.is_collect_state, config=config)


    trainer_p2 = RLTrainer(env, p2, p2_old, config)
    #rollout_p2 = rollouts.traj_segment_generator_rl(env, p2, stochastic=not config.is_collect_state, config=config)
    rollout_p2 = rollouts.traj_segment_coart(env, [p1, p2], stochastic=not config.is_collect_state, config=config)

    p1c_path = load_model(p1_path, p1_vars)
    p2c_path = load_model(p2_path, p2_vars)
    trainer_p2.evaluate(rollout_p2, ckpt_num=p2c_path.split('/')[-1])
    
    config.num_rollouts = initial_rollouts # NOTE : set it back
