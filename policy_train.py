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
def get_obs_label(train):
    train_obs = train[:,0:42]
    train_labels = train[:,42:44]
    train_labels[:, 0] = train_labels[:,0]/4
    train_labels[:, 1] = train_labels[:,1]/2.5
    return train_obs, train_labels


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
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    test = df[~msk]
    print(len(train))
    print(num_data_points)

    n_epoch = 1000
    graph = model.sess.graph
    with model.sess as sess:
        target_ph = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        train_obs, train_labels = get_obs_label(train)
        test_obs, test_labels = get_obs_label(test)
        #print(obs)
        #print(labels)
        tarin_obs = train_obs.reshape((-1,) + model.observation_space.shape) 
        test_obs = test_obs.reshape((-1,) + model.observation_space.shape) 
        #loss = tf.keras.losses.mean_squared_error(model.act_model.deterministic_action, target_ph)
        loss = tf.reduce_mean(tf.squared_difference(model.act_model.deterministic_action, target_ph))
        #loss = tf.nn.l2_loss(model.act_model.deterministic_action - target_ph)/num_data_points
        optim = tf.train.AdamOptimizer(learning_rate=0.0001, name='Adamopt')
        train_op = optim.minimize(loss)
        sess.run(tf.initialize_variables((optim.variables())))
        train_loss = []
        test_loss = []
        #action = model.sess.run(model.act_model.deterministic_action,{model.act_model.obs_ph: train_obs})
        action = model.sess.run(model.act_model.deterministic_action,{model.act_model.obs_ph: test_obs})
        _, loss1 = sess.run([train_op,loss], {target_ph: train_labels, model.act_model.obs_ph: train_obs})
       # print(loss1)
        for i in range(0,n_epoch):
            _, loss2 = sess.run([train_op ,loss], {target_ph: train_labels, model.act_model.obs_ph: train_obs})
            train_loss.append(loss2)
            loss3 = sess.run([loss], {target_ph: test_labels, model.act_model.obs_ph: test_obs})
            test_loss.append(loss3)
            if i % 100 == 0:
                print(f"Iteration: {i}, Traninig Loss: {loss2}")
                print(f"Iteration: {i}, Test Loss: {loss3}")
                print("----------------------------------------")
            #print(loss2)
        action1 = model.sess.run(model.act_model.deterministic_action,{model.act_model.obs_ph: test_obs})
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        plt.figure(2)
        plt.plot(action[:,0]*4,'r', label="Linear Velocity Prediction")
        plt.plot(test_labels[:,0]*4, 'g', label="Linear Velocity Label")
        plt.ylabel("Velocity in m/s")
        plt.legend()
        plt.title("Linear velocity comaparision on test data before training")
        plt.grid()
        plt.figure(3)
        plt.plot(action[:,1]*2.5,'r', label="Angular Velocity Prediction")
        plt.plot(test_labels[:,1]*2.5, 'g', label="Angular Velocity Label")
        plt.ylabel("Angular velocity in rad/s")
        plt.legend()
        plt.title("Angualr velocity comaparision on test data before training")
        plt.grid()
        plt.figure(4)
        plt.plot(action1[:,0]*4,'r', label="Linear Velocity Prediction")
        plt.plot(test_labels[:,0]*4, 'g', label="Linear Velocity label")
        plt.ylabel("Velocity in m/s")
        plt.legend()
        plt.title("Linear velocity comaparision on test data after training")
        plt.grid()
        plt.figure(5)
        plt.plot(action1[:,1]*2.5,'r', label="Angular Velocity Prediction")
        plt.plot(test_labels[:,1]*2.5, 'g', label="Angular Velocity Prediction")
        plt.ylabel("Angular velocity in rad/s")
        plt.legend()
        plt.title("Angular velocity comaparision on test data after training")
        plt.grid()
        plt.figure(6)
        plt.plot(train_loss, 'r', label="Training Loss")
        plt.plot(test_loss, 'g', label="Test Loss")
        plt.legend()
        plt.title("Training Loss vs Test Loss")
        plt.grid()
        model.save("./policy/after_train")

    plt.grid()
    plt.show()
    writer.flush()
    writer.close()

if __name__ == '__main__':
        main()