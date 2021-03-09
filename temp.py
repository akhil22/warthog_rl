from __future__ import print_function
import tensorflow as tf
import numpy as np
with tf.Session() as sess:
    v1 = tf.placeholder(dtype=tf.float32, shape=[None,2])
    v2 = tf.placeholder(dtype=tf.float32, shape=[None,2])
    v3 = tf.Variable([[0., 1.]], dtype=tf.float32)
    sess.run(tf.initialize_variables([v3]))

    ad = tf.compat.v1.train.AdamOptimizer(0.1, name='Ad')
    loss = tf.reduce_mean(v1*v3 - v2)
    train_op = ad.minimize(loss)
    print(ad.variables())
    sess.run(tf.global_variables_initializer())
    sess.run(train_op, {v1:np.array([[0.,1.]]), v2: np.array([[1., 2.]])})
    writer = tf.summary.FileWriter('./graphs2', sess.graph)
    #sess.run(tf.global_variables_initializer())
    #print(ad.variables())
writer.flush()
writer.close()
