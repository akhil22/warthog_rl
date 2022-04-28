import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines import PPO2
from env.WarthogEnvAirSim import WarthogEnv
from matplotlib import pyplot as plt

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = WarthogEnv('unity_remote.txt')
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

if __name__ == '__main__':
    env_id = "CartPole-v1"
    num_cpu = 1  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    plt.pause(2)
    fname = './policy/vel_airsim'

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)
    for i in range(0,10):
        if i == 0:
            model = PPO2('MlpPolicy', env, verbose=1)
            model.learn(total_timesteps=1000000)
            model.save(f'{fname}0')
        else :
            model1 = PPO2('MlpPolicy', env, verbose=1)
            #for learning uncomment
            model = PPO2('MlpPolicy', env, verbose=1)
            # model.load('./first_pytorch_multiplication_reward.zip')
            model = PPO2.load(f'{fname}{i-1}')
            model.env = model1.env
            model.learn(total_timesteps=1000000)
            model.save(f'{fname}{i}')

