import os
import h5py
import tensorflow as tf
import baselines.common.tf_util as U
import logging

from baselines.common.atari_wrappers import TransitionEnvWrapper

def make_env(env_name, config=None):
    import gym
    env = gym.make(env_name)
    gym.logger.setLevel(logging.WARN)
    if config:
        try:
            env.unwrapped.set_environment_config(config.env_args)
            gym.logger.info("Set the configuration to the environment: "
                            "{}".format(config.env_args))
        except:
            gym.logger.info("Can't set the configuration to the environment! "
                            "Use the default setting instead of "
                            "({})".format(config.env_args))

        assert env.spec.max_episode_steps <= config.num_rollouts, \
            '--num_rollouts ({}) should be larger than a game length ({})'.format(
                config.num_rollouts, env.spec.max_episode_steps)

    env = TransitionEnvWrapper(env)
    return env

def load_model(load_model_path, var_list=None):
    if os.path.isdir(load_model_path):
        ckpt_path = tf.train.latest_checkpoint(load_model_path)
    else:
        ckpt_path = load_model_path
        #U.initialize()
        #U.save_state(ckpt_path, var_list)
    if ckpt_path:
        U.load_state(ckpt_path, var_list)
    return ckpt_path

def load_buffers(proximity_predictors, ckpt_path):
    if proximity_predictors:
        buffer_path = ckpt_path + '.hdf5'
        if os.path.exists(buffer_path):
            logger.info('Load buffers from {}'.format(buffer_path))
            with h5py.File(buffer_path, 'r') as buffer_file:
                for p in proximity_predictors:
                    success_obs = buffer_file[p.env_name]['success'].value
                    fail_obs = buffer_file[p.env_name]['fail'].value
                    if success_obs.shape[0]:
                        p.success_buffer.add(success_obs)
                    if fail_obs.shape[0]:
                        p.fail_buffer.add(fail_obs)
                    logger.info('Load buffers for {}. success states ({})  fail states ({})'.format(
                        p.env_name, success_obs.shape[0], fail_obs.shape[0]))
        else:
            logger.warn('No buffers are available at {}'.format(buffer_path))
