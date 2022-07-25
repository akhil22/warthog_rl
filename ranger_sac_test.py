#!/usr/bin/env python
# coding: utf-8

# In[1]:

import gym

from stable_baselines.common.policies import MlpPolicy
#from stable_baselines.common import make_vec_env
from stable_baselines import SAC as PPO2
from matplotlib import pyplot as plt
from env.RangerEnv_vis import RangerEnv
import time
import numpy as np
import sys

# In[4]:


def main():
    #env = WarthogEnv('unity_remote.txt')
    #env = RangerEnv('sim_remote_waypoint.txt')
    env = RangerEnv('ranger_waypoints2.csv')
    #env = WarthogEnv('real_remote_waypoints.txt')
    plt.pause(2)
    model1 = PPO2('MlpPolicy', env, verbose=1)
    #model = PPO2('MlpPolicy', env, verbose=1)
    #model.save('./policy/zero_train')
    model = PPO2.load('./policy/ranger3_sac_weight_action_penalty6')
    #model = PPO2.load('./policy/after_train_const')
    #model = PPO2.load('./policy/after_train_const_delay')
    #model = PPO2.load('./policy/combine_trained')
    #model = PPO2.load(sys.argv[1])
    #model = PPO2.load('./policy/after_train_const_zero')
    #model = PPO2.load('./policy/zero_train')
    #model = PPO2.load('./policy/real_train_const_zero')
    #model.env = model1.env
    act1 = []
    act2 = []
    #action = [0., 0., 0.]
    reward = 0
    #envg = model.get_env()
    #action[0] = 0.1
    #action[1] = 0.0
    #action[2] = 0.0
    #obs = env.reset()
    #env.veh.SetInitialPosition(100,100,0)
    #env.veh.SetInitialHeading(0)
    '''
    obs = env.reset()
    obs, reward, done, info = env.step(action)
    env.veh.Update(env.mavs_env, 0.0, 0.0, 1.0, 0.000001)
    print(env.veh.GetPosition())
    env.step(action)
    env.render()
    plt.pause(3)
    obs = env.reset()
    #env.veh.SetInitialPosition(100,100,0)
    #env.veh.SetInitialHeading(0)
    obs, reward, done, info = env.step(action)
    env.veh.Update(env.mavs_env, 0.0, 0.0, 1.0, 0.000001)
    print(env.veh.GetPosition())
    env.step(action)
    env.render()
    plt.pause(3)
    #env.veh.SetInitialPosition(100,100,0)
    #env.veh.SetInitialHeading(0) 
    obs = env.reset()
    #obs, reward, done, info = env.step(action)
    env.veh.Update(env.mavs_env, 0.0, 0.0, 1.0, 0.000001)
    print(env.veh.GetPosition())
    env.step(action)
    env.render()'''
    #t1 = time.time()
    obs = env.reset()
    for i in range(30000):
        #  t2 = time.time()
        action, _states = model.predict(obs, deterministic=False)
        #action, _states = model.predict(obs, deterministic = True)
        # print(action)
        #act1.append(np.clip(action[0], 0 ,1)*4)
        #act2.append(np.clip(action[1], -1 ,1)*2.5)
        #act2.append(reward)
        #action[0] = np.clip(action[0], 0, 1)*4
        #action[1] = np.clip(action[1], -1, 1)*2.5
        #action[0] = 0.1
        #action[1] = 0.0
        #kkkkkkkkkkaction[2] = 0.3
        obs, reward, done, info = env.step(action)
        #print(t2-t1)
        #if t2 -t1 < 0.3:
        # time.sleep(0.3 - (t2-t1))
        #t1 = t2
        #print(action)
        env.render()
        #time.sleep(2)
        if done:
            obs = env.reset()


# In[6]:

    plt.figure(2)
    plt.plot(act1)

    # In[7]:

    plt.figure(3)
    plt.plot(act2)
    plt.show()
if __name__ == '__main__':
    main()

# In[ ]: