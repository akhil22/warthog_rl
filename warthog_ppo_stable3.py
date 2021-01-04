import gym
from env.WarthogEnv import WarthogEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from stable_baselines3 import PPO
from typing import Callable

def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = WarthogEnv('unity_remote.txt')
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init
def main():
    env_id = "CartPole-v1"
    num_cpu = 20  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    for i in range(0,10):
        if i == 0:
            model = PPO('MlpPolicy', env, verbose=1)
            model.learn(total_timesteps=1e6)
            model.save("samestart0")
        else :
            model1 = PPO('MlpPolicy', env, verbose=1)
            #for learning uncomment
            model = PPO('MlpPolicy', env, verbose=1)
            # model.load('./first_pytorch_multiplication_reward.zip')
            model = PPO.load(f'samestart{i-1}')
            model.env = model1.env
            model.learn(total_timesteps=1e6)
            model.save(f'samestart{i}')
        #env = WarthogEnv('unity_remote.txt')
    #env.reset()
    #for i in range(0,5000):
       # action = [[0.5,0.1]]*num_cpu
        #action = [0.5,0.1]
       # obs, reward, done, info = env.step(action)
        #env.render()
        #x.append(info[0][0])
        #y.append(info[0][1])
        #if done:
            #print("resetting")
         #   env.reset()


if __name__ == '__main__':
    main()
