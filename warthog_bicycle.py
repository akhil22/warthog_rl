import gym
from env.WarthogEnv import WarthogEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
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
        env = WarthogEnv('unity_remote.txt', None)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init
def main():
    env_id = "CartPole-v1"
    num_cpu = 1  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    fname = 'temp_policy/stable3_maching_tf_params_ppo'
    for i in range(0,100):
        if i == 0:
            #model = PPO('MlpPolicy', env, verbose=1)
            model = PPO('MlpPolicy', env, verbose=1, n_steps=128, ent_coef=0.01, learning_rate=0.00025, batch_size=4, n_epochs=4)
            #model = PPO('MlpPolicy', env, verbose=1, n_steps=80, ent_coef=0.01, learning_rate=0.00003, batch_size=4000, n_epochs=80)
            model.learn(total_timesteps=1e5)
            model.save(f'{fname}0')
        else :
            #model1 = PPO('MlpPolicy', env, verbose=1)
            #model1 = PPO('MlpPolicy', env, verbose=1, n_steps=4000, ent_coef=0.01, learning_rate=0.00003, batch_size=4000, n_epochs=80)
            model = PPO('MlpPolicy', env, verbose=1, n_steps=128, ent_coef=0.01, learning_rate=0.00025, batch_size=4, n_epochs=4)
            model1 = PPO('MlpPolicy', env, verbose=1, n_steps=128, ent_coef=0.01, learning_rate=0.00025, batch_size=4, n_epochs=4)
            #model1 = PPO('MlpPolicy', env, verbose=1, n_steps=4000, ent_coef=0.01, learning_rate=0.00003, batch_size=4000, n_epochs=80)
            #for learning uncomment
            #model = PPO('MlpPolicy', env, verbose=1)
            #model = PPO('MlpPolicy', env, verbose=1, n_steps=4000, ent_coef=0.01, learning_rate=0.00003, batch_size=4000, n_epochs=80)
            #model = PPO('MlpPolicy', env, verbose=1, n_steps=4000, ent_coef=0.01, learning_rate=0.00003, batch_size=4000, n_epochs=80)
            #model = PPO('MlpPolicy', env, verbose=1, n_steps=128, ent_coef=0.01, learning_rate=0.00025, batch_size=4, n_epochs=4)
            # model.load('./first_pytorch_multiplication_reward.zip')
            model = PPO.load(f'{fname}{i-1}')
            model.env = model1.env
            model.learn(total_timesteps=1e5)
            model.save(f'{fname}{i}')
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
