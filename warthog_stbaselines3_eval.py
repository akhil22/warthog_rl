#!/usr/bin/env python
# coding: utf-8


import gym

#from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from matplotlib import pyplot as plt
#from env.WarthogEnvAirSim import WarthogEnv
from env.WarthogEnv import WarthogEnv
import time
import numpy as np
import sys
from tqdm import tqdm


def main():
    #env = WarthogEnv('unity_remote.txt')
    env = WarthogEnv('sim_remote_waypoint.txt',None)
    #env = WarthogEnv('real_remote_waypoints.txt')
    #plt.pause(2)
    model1 = PPO('MlpPolicy', env, verbose=1)
    model = PPO('MlpPolicy', env, verbose=1)
    num_runs = 10
    num_steps = 3000
    policy_file = sys.argv[1]
    fig_file_ext = sys.argv[2]
    avg_rew = np.zeros([num_runs,num_steps])
    #model.save('./policy/zero_train')
    #model = PPO2.load('./policy/vel_weight8_stable8')
    #model = PPO2.load('./policy/vel_airsim_test_final_6xfast3')
    #model = PPO2.load('./policy/kinematic_sup0_after_corr_train_200')
    #model = PPO2.load('./policy/kinematic_sup0_after_corr_train_500k_1M200')
    model = PPO.load(f'./temp_policy/{policy_file}')
    #model = PPO2.load('./policy/after_train_const')
    #model = PPO2.load('./policy/after_train_const_delay')
    #model = PPO2.load('./policy/combine_trained')
    #model = PPO2.load(sys.argv[1])
    #model = PPO2.load('./policy/after_train_const_zero')
    #model = PPO2.load('./policy/zero_train')
    #model = PPO2.load('./policy/real_train_const_zero')
    model.env = model1.env
    act1 = []
    act2 = []
    reward = 0
    #envg = model.get_env()
    obs = env.reset()
    #t1 = time.time()
    for j in tqdm(range(0, num_runs)):
        reward = 0
        rewarr = [0]

        for i in range(3000):
            action, _states = model.predict(obs, deterministic=True)
            act1.append(np.clip(action[0], 0, 1) * 4)
            act2.append(np.clip(action[1], -1, 1) * 2.5)
            obs, reward, done, info = env.step(action)
            rewarr.append(reward + rewarr[-1])
            avg_rew[j][i] = reward+rewarr[-1]
            #env.render()
            if done:
                obs = env.reset()



    fig = plt.figure(2)
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

