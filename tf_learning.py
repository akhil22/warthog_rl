import tensorflow as tf

writer = tf.summary.FileWriter('./graphs')

accuracy = [0.1, 0.4, 0.6, 0.8, 0.9, 0.95]
acc_var = tf.Variable(0, dtype=tf.float32)
acc_sum = tf.summary.scalar('Accuracy', acc_var)

sess = tf.Session()

for step, acc in enumerate(accuracy):
    sess.run(acc_var.assign(acc))
    writer.add_summary(sess.run(acc_sum), step)

writer.flush()
writer.close()
