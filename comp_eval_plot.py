#!/usr/bin/env python
# coding: utf-8

# In[1]:

import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from matplotlib import pyplot as plt
#from env.WarthogEnvAirSim import WarthogEnv
from env.WarthogEnv import WarthogEnv
import time
import numpy as np
import sys
from tqdm import tqdm

# In[4]:


def main():
    #env = WarthogEnv('unity_remote.txt')
    fig = plt.figure(3)
    for i in range (len(sys.argv)-2):
        b = np.loadtxt(f'./{sys.argv[i+1]}', dtype=float) 
        plt.plot(b, label=f'{sys.argv[i+1]}')
    plt.legend()
    plt.title(f"Average cumulative Reward vs time for various policies")
    plt.xlabel("time step")
    plt.ylabel("Average cumulative reward over 10 runs")
    fig_file= sys.argv[len(sys.argv)-1]
    fig.savefig(fig_file)
    plt.close(fig)

# In[6]:

    #plt.figure(2)
    #plt.plot(act1)

    # In[7]:kkkkkkkkk
    #plt.savefig(f"./{policy_file}_reward_plot.png")
    #plt.savefig("temp2.png")
    #fig.savefig(fig_file)
#    print(rewarr)
if __name__ == '__main__':
    main()

#
