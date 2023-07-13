#!/usr/bin/env python
# coding: utf-8


import gym

from matplotlib import pyplot as plt
#from env.WarthogEnvAirSim import WarthogEnv
from env.WarthogEnv import WarthogEnv
import time
import numpy as np
import sys



def main():
    #env = WarthogEnv('unity_remote.txt')
    env = WarthogEnv('sim_remote_waypoint.txt',None)
    plt.pause(2)
    #env = WarthogEnv('real_remote_waypoints.txt')
    act1 = []
    act2 = []
    reward = 0
    #envg = model.get_env()
    obs = env.reset()
    #t1 = time.time()
    for i in range(3000):
        #  t2 = time.time()
        #action, _states = model.predict(obs, deterministic=False)
        #action, _states = model.predict(obs, deterministic=True)
        # print(action)
        act1.append(np.clip(0.5, 0, 1) * 4)
        act2.append(np.clip(0.01, -1, 1) * 2.5)
        #act2.append(reward)
        #action[0] = np.clip(action[0], 0, 1)*4
        #action[1] = np.clip(action[1], -1, 1)*2.5
        obs, reward, done, info = env.step([2.0, 0.2])
        #print(t2-t1)
        #if t2 -t1 < 0.3:
        # time.sleep(0.3 - (t2-t1))
        #t1 = t2
        #print(action)
        env.render()
        #time.sleep(2)
        if done:
            obs = env.reset()



    plt.figure(2)
    plt.plot(act1)


    plt.figure(3)
    plt.plot(act2)
    plt.show()
if __name__ == '__main__':
    main()

