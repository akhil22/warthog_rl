""" Retrain the RL Policy on with collected data

Retrain the polic obtained from PPO with 
collected simulated or real data
"""
import tensorflow as tf

import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from matplotlib import pyplot as plt
from env.WarthogEnv import WarthogEnv
import sys
import numpy as np
import pandas as pd
from ast import literal_eval

# In[4]:

def main():
    """ Read the ppo policy and collected data
    and train the policy further on this data

    Args:
        sys.argv[1]: ppo policy file for retraining
        sys.argv[2]: training data file for policy
     
    """
    env = WarthogEnv('unity_remote.txt')
    obs = np.array(env.reset())
    model = PPO2('MlpPolicy', env, verbose=1)
    model = PPO2.load('./policy/vel_weight8_stable9')
    model = PPO2.load(sys.argv[1])
    df = pd.read_csv(sys.argv[2])
    num_data_points = len(df.index)
    print(num_data_points)
    #action3, _states2 = model.predict(obs, deterministic = True)
    #x = tf.placeholder(dtype=tf.float32, name='out2')
    #outaction = tf.Variable(0.1, dtype=np.float32)
    graph = model.sess.graph
    with model.sess as sess:
        #init = tf.global_variables_initializer()
        #sess.run(init)
        #op_to_restore = graph.get_tensor_by_name("output/add:0")
        #def1 = tf.multiply(1.0, op_to_restore)
        target_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        for i in range(0, num_data_points):
            row = df.iloc[i]
            obs = np.array(row[0:42])
            print(obs)
            obs = obs.reshape((-1,) + model.observation_space.shape) 
            label_action = np.array(row[42:44])
            #init_new_vars_op = tf.initialize_variables([x])
            #init_new_vars_op = tf.initialize_variables([target])
            #sess.run(init_new_vars_op)
            #def1 = tf.multiply(1.0, model.act_model.deterministic_action)
            # def1 = tf.multiply(model.act_model.deterministic_action, x)
            action = model.sess.run(model.act_model.deterministic_action,{model.act_model.obs_ph: obs})
            #action2, _states = model.predict(obs2, deterministic = True)
            action[0][0] = np.clip(0, 1, action[0][0])*4 
            action[0][1] = np.clip(-1, 1, action[0][1])*2.5
            print(action)
            print(label_action)
            print("\n")
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        #sess.run(def1)

    writer.flush()
    writer.close()

if __name__ == '__main__':
        main()
