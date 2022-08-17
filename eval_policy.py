#!/usr/bin/env python
# coding: utf-8

# In[1]:

import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from matplotlib import pyplot as plt
#from env.WarthogEnvAirSim import WarthogEnv
from env.WarthogEnvSup import WarthogEnv
import time
import numpy as np
import sys
from tqdm import tqdm

# In[4]:


def main():
    #env = WarthogEnv('unity_remote.txt')
    num_runs = 10
    num_steps = 3000
    avg_rew = np.zeros([num_runs,num_steps])
    env = WarthogEnv('sim_remote_waypoint.txt', 'None')
    #env = WarthogEnv('temp_way.txt', 'None')
    model1 = PPO2('MlpPolicy', env, verbose=0)
    model = PPO2('MlpPolicy', env, verbose=0)
    policy_file = sys.argv[1]
    fig_file_ext = sys.argv[2]
    model = PPO2.load(f"./policy/{policy_file}")
    model.env = model1.env
    #plt.pause(2)
    for j in tqdm(range(0,num_runs)):
#        print(j)
    #env = WarthogEnv('real_remote_waypoints.txt')
    #model.save('./policy/zero_train')
    #model = PPO2.load('./policy/vel_weight8_stable8')
    #plot_files = sys.argv[2]
    #model = PPO2.load('./policy/vel_airsim_test_final_6xfast3')
    #model = PPO2.load('./policy/after_train_const')
    #model = PPO2.load('./policy/after_train_const_delay')
    #model = PPO2.load('./policy/combine_trained')
    #model = PPO2.load(sys.argv[1])
    #model = PPO2.load('./policy/after_train_const_zero')
    #model = PPO2.load('./policy/zero_train')
    #model = PPO2.load('./policy/real_train_const_zero')
        act1 = []
        act2 = []
        reward = 0
        rewarr = [0]
    #envg = model.get_env()
        obs = env.reset()
    #t1 = time.time()
        for i in range(num_steps):
#            print(i)
        #  t2 = time.time()
        #action, _states = model.predict(obs, deterministic=False)
            action, _states = model.predict(obs, deterministic=True)
        # print(action)
            act1.append(np.clip(action[0], 0, 1) * 4)
            act2.append(np.clip(action[1], -1, 1) * 2.5)
        #act2.append(reward)
        #action[0] = np.clip(action[0], 0, 1)*4
        #action[1] = np.clip(action[1], -1, 1)*2.5
            obs, reward, done, info = env.step(action)
            rewarr.append(reward + rewarr[-1])
            avg_rew[j][i] = reward+rewarr[-1]
            #avg_rew[j][i] = reward

        #print(t2-t1)
        #if t2 -t1 < 0.3:
        # time.sleep(0.3 - (t2-t1))
        #t1 = t2
        #print(action)
#            env.render()
        #time.sleep(2)
            if done:
                obs = env.reset()



    plt.figure(2)
    plt.plot(act1)
    plt.figure(3)
    plt.plot(act2)
    plt.show()

    # In[7]:kkkkkkkkk

    fig = plt.figure(4)
    avg_rew = avg_rew.mean(axis=0)
    np.savetxt(f'avg_rew_{policy_file}', avg_rew, fmt='%f')
    plt.plot(avg_rew)
    plt.title(f"Cummulative Reward vs time for {policy_file}")
    plt.xlabel("time step")
    plt.ylabel("reward")
    plt.ylim([0, 17000])
    fig_file=f"{fig_file_ext}/{policy_file}_reward_plot.jpg"
    #plt.savefig(f"./{policy_file}_reward_plot.png")
    #plt.savefig("temp2.png")
    #fig.savefig(fig_file)
    fig.savefig(fig_file)
    plt.close(fig)
#    print(rewarr)
if __name__ == '__main__':
    main()

