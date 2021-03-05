import tensorflow as tf

import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from matplotlib import pyplot as plt
from env.WarthogEnv import WarthogEnv
import numpy as np


# In[4]:


env = WarthogEnv('unity_remote.txt')
obs = env.reset()
obs2 = obs
obs = np.array(obs)



# In[5]:


model1 = PPO2('MlpPolicy', env, verbose=1)
model = PPO2('MlpPolicy', env, verbose=1)
model = PPO2.load('./policy/vel_weight8_stable9')
model.env = model1.env
action2, _states = model.predict(obs, deterministic = True)
#action3, _states2 = model.predict(obs, deterministic = True)
#x = tf.placeholder(dtype=tf.float32, name='out2')
#outaction = tf.Variable(0.1, dtype=np.float32)
graph = model.sess.graph
with model.sess as sess:
    #init = tf.global_variables_initializer()
    #sess.run(init)
    #op_to_restore = graph.get_tensor_by_name("output/add:0")
    #def1 = tf.multiply(1.0, op_to_restore)
    x = tf.Variable(1.0, dtype = tf.float32)
    init_new_vars_op = tf.initialize_variables([x])
    sess.run(init_new_vars_op)
    #def1 = tf.multiply(1.0, model.act_model.deterministic_action)
    def1 = tf.multiply(model.act_model.deterministic_action, x)
    obs = obs.reshape((-1,) + model.observation_space.shape) 
    #action = model.sess.run(model.act_model.deterministic_action, {model.act_model.obs_ph: obs})
    action = model.sess.run(def1,{model.act_model.obs_ph: obs})
    action3, _states = model.predict(obs, deterministic = True)
    #action2, _states = model.predict(obs2, deterministic = True)
    print(action, action2, action3)
    writer = tf.summary.FileWriter('./graphs', sess.graph)
#sess.run(def1)

writer.flush()
writer.close()
