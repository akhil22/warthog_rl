#!/usr/bin/env python
# coding: utf-8

# In[1]:

import gym
from stable_baselines.common import make_vec_env
from matplotlib import pyplot as plt
#from env.WarthogEnvAirSim import WarthogEnv
from env.WarthogEnv import WarthogEnv
import numpy as np
import sys

# In[4]:


def main():
    #env = WarthogEnv('unity_remote.txt')
    env = WarthogEnv('sim_remote_waypoint.txt', 'manual_pose.csv')
    start_v = 1.
    start_w = -2.5
    vel_samples = 5
    w_samples = 40
    #plt.pause(2)
    for i in range(0,5 - int(start_v)):
        start_w = -4.
        for j in range(0,40):
            t = 0 
            while t < 100:
                action = [start_v/4.0, start_w/2.5]
                print(action)
                obs, reward, done, info = env.step(action)                #print(t)
                t = t+1
                #print(t)
                #env.render()
            start_w = start_w + 0.2
        start_v = start_v + 1
# In[6]:

    #plt.figure(2)
    #plt.plot(act1)

    # In[7]:kkkkkkkkk

#    print(rewarr)
if __name__ == '__main__':
    main()

# In[ ]:
