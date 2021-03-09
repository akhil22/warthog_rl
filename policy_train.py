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
from matplotlib import pyplot as plt

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
    df = df.to_numpy()
    print(num_data_points)
    n_epoch = 10
    #action3, _states2 = model.predict(obs, deterministic = True)
    #x = tf.placeholder(dtype=tf.float32, name='out2')
    #outaction = tf.Variable(0.1, dtype=np.float32)
    graph = model.sess.graph
    with model.sess as sess:
        target_ph = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        obs = df[:,0:42]
        labels = df[:,42:44]
        labels[:, 0] = labels[:,0]/4
        labels[:, 1] = labels[:,1]/2.5
        print(obs)
        print(labels)
        obs = obs.reshape((-1,) + model.observation_space.shape) 
        #loss = tf.keras.losses.mean_squared_error(model.act_model.deterministic_action, target_ph)
        #loss = tf.reduce_mean(tf.squared_difference(model.act_model.deterministic_action, target_ph))
        loss = tf.nn.l2_loss(model.act_model.deterministic_action - target_ph)/num_data_points
        optim = tf.train.AdamOptimizer(learning_rate=0.0001, name='Adamopt')
        train_op = optim.minimize(loss)
        sess.run(tf.initialize_variables((optim.variables())))
        action = model.sess.run(model.act_model.deterministic_action,{model.act_model.obs_ph: obs})
        _, loss1 = sess.run([train_op,loss], {target_ph: labels, model.act_model.obs_ph: obs})
        print(loss1)
        for i in range(0,100):
            _, loss2 = sess.run([train_op ,loss], {target_ph: labels, model.act_model.obs_ph: obs})
            print(loss2)
        action1 = model.sess.run(model.act_model.deterministic_action,{model.act_model.obs_ph: obs})
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        plt.figure(2)
        plt.plot(action[:,0]*4,'r')
        plt.plot(labels[:,0]*4, 'g')
        plt.figure(3)
        plt.plot(action[:,1]*2.5,'r')
        plt.plot(labels[:,1]*2.5, 'g')
        plt.figure(4)
        plt.plot(action1[:,0]*4,'r')
        plt.plot(labels[:,0]*4, 'g')
        plt.figure(5)
        plt.plot(action1[:,1]*2.5,'r')
        plt.plot(labels[:,1]*2.5, 'g')

    plt.show()
    writer.flush()
    writer.close()

if __name__ == '__main__':
        main()
