import gym
from env.WarthogEnv import WarthogEnv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise 
from typing import Callable
from stable_baselines3.ddpg.policies import MlpPolicy

def main():
    # Create the vectorized environment
    env = WarthogEnv('unity_remote.txt')
    plt.pause(1)
    n_actions = env.action_space.shape[-1]
    print(n_actions)
    param_noise = None
    #action_noise = OrnsteinUhlenbeckActionNoise(mean = np.zeros(n_actions), sigma=float(0.5)*np.ones(n_actions))
    action_noise = NormalActionNoise(mean = np.zeros(n_actions), sigma=float(0.5)*np.ones(n_actions))
    fname='ddpg_model'
    for i in range(0,10):
        if i == 0:
            model = DDPG(MlpPolicy, env, action_noise=action_noise, verbose=1)
            model.learn(total_timesteps=1e5)
            model.save(f'{fname}{i}')
        else :
            model1 = DDPG(MlpPolicy, env, action_noise=action_noise, verbose=1)
            #for learning uncomment
            model = DDPG(MlpPolicy, env, action_noise=action_noise, verbose=1)
            # model.load('./first_pytorch_multiplication_reward.zip')
            model = DDPG.load(f'{fname}{i-1}')
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
