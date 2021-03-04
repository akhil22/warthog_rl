#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from matplotlib import pyplot as plt
from env.WarthogEnv import WarthogEnv
import numpy as np


# In[4]:


env = WarthogEnv('unity_remote.txt')
plt.pause(2)


# In[5]:


model1 = PPO2('MlpPolicy', env, verbose=1)
model = PPO2('MlpPolicy', env, verbose=1)
model = PPO2.load('./policy/vel_weight8_stable9')
model.env = model1.env
act1 = []
act2 = []
reward = 0
    #envg = model.get_env()
obs = env.reset()
    #t1 = time.time()
for i in range(5000):
      #  t2 = time.time()
        #action, _states = model.predict(obs, deterministic=False) 
    action, _states = model.predict(obs)
       # print(action)
    act1.append(np.clip(action[0], 0 ,1)*4)
    act2.append(np.clip(action[1], -1 ,1)*2.5)
        #act2.append(reward) 
        #action[0] = np.clip(action[0], 0, 1)*4
        #action[1] = np.clip(action[1], -1, 1)*2.5
    obs, reward, done, info = env.step(action)
        #print(t2-t1)
        #if t2 -t1 < 0.3:
           # time.sleep(0.3 - (t2-t1))
        #t1 = t2
    #print(action)
    env.render()
    if done:
        obs = env.reset()


# In[6]:

plt.figure(2)
plt.plot(act1)


# In[7]:


plt.figure(3)
plt.plot(act2)
plt.show()


# In[ ]:



