#!/usr/bin/env python
# coding: utf-8


import gym

#from stable_baselines3.common.policies import MlpPolicy
#from env.WarthogEnvAirSim import WarthogEnv
from env.WarthogEnv import WarthogEnv
import time
import numpy as np
import sys
import torch
from torch import nn
from torch.distributions import Normal
import matplotlib.pyplot as plt
from tqdm import tqdm


class PolicyNetworkGauss(nn.Module):
    def __init__(self, obs_dimension, sizes, action_dimension, act=nn.ReLU):
        super(PolicyNetworkGauss, self).__init__()
        sizes = [obs_dimension] + sizes + [action_dimension]
        out_activation = nn.Identity
        self.layers = []
        for j in range(0, len(sizes) - 1):
            act_l = act if j < len(sizes) - 2 else out_activation
            self.layers += [nn.Linear(sizes[j], sizes[j + 1]), act_l()]
        self.mu = nn.Sequential(*self.layers)
        log_std = -0.5 * np.ones(action_dimension, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def forward(self, x):
        mean = self.mu(x)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        return dist




def main():
    num_runs = 10
    num_steps = 3000
    policy_file = sys.argv[1]
    fig_file_ext = sys.argv[2]
    avg_rew = np.zeros([num_runs,num_steps])
    #env = WarthogEnv('unity_remote.txt')
    env = WarthogEnv('sim_remote_waypoint.txt', None)
    #env = WarthogEnv('real_remote_waypoints.txt')
    #plt.pause(2)
    model = torch.load(f'./temp_policy/tan_h/{policy_file}')
    #model.save('./policy/zero_train')
    #model = PPO2.load('./policy/vel_weight8_stable8')
    #model = PPO2.load('./policy/vel_airsim_test_final_6xfast3')
    #model = PPO2.load('./policy/kinematic_sup0_after_corr_train_200')
    #model = PPO2.load('./policy/kinematic_sup0_after_corr_train_500k_1M200')
    #model = PPO2.load('./policy/after_train_const')
    #model = PPO2.load('./policy/after_train_const_delay')
    #model = PPO2.load('./policy/combine_trained')
    #model = PPO2.load(sys.argv[1])
    #model = PPO2.load('./policy/after_train_const_zero')
    #model = PPO2.load('./policy/zero_train')
    #model = PPO2.load('./policy/real_train_const_zero')
    act1 = []
    act2 = []
    #envg = model.get_env()
    obs = env.reset()
    #t1 = time.time()
    for j in tqdm(range(0, num_runs)):
        reward = 0
        rewarr = [0]
        for i in range(num_steps):
        #  t2 = time.time()
        #action, _states = model.predict(obs, deterministic=False)
            with torch.no_grad():
                m = model(torch.as_tensor(obs, dtype=torch.float32))
                action = m.sample()
        # print(action)
            act1.append(np.clip(action[0], 0, 1) * 4)
            act2.append(np.clip(action[1], -1, 1) * 2.5)
        #act2.append(reward)
        #action[0] = np.clip(action[0], 0, 1)*4
        #action[1] = np.clip(action[1], -1, 1)*2.5
            obs, reward, done, info = env.step(action)
            rewarr.append(reward + rewarr[-1])
            avg_rew[j][i] = reward + rewarr[-1]
        #print(t2-t1)
        #if t2 -t1 < 0.3:
        # time.sleep(0.3 - (t2-t1))
        #t1 = t2
        #print(action)
            #env.render()
        #time.sleep(2)
            if done:
                obs = env.reset()


    plt.figure(2)
    plt.plot(act1)

    # In[7]:

    plt.figure(3)
    plt.plot(act2)
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
if __name__ == '__main__':
    main()
