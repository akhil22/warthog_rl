from __future__ import print_function
import os
import random
import skimage.io as im
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
import tensorflow as tf
def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    # print(directories)
    for step, d in enumerate(directories):
        label_directory = os.path.join(data_directory, d)
        # print(label_directory)
        file_names = [os.path.join(label_directory, f) for f in os.listdir(label_directory) if f.endswith(".ppm")]
        for f in file_names:
            images.append(im.imread(f))
            labels.append(int(d))

    return images, labels
    print(directories)
ROOT_DIR='/home/sai/Downloads/BTSC'
train_data_dir = os.path.join(ROOT_DIR, "Training")
test_data_dir = os.path.join(ROOT_DIR, "Testing")

images, labels = load_data(train_data_dir)
images = np.array(images)
#labels= np.array(labels)
print(images.itemsize)
print(images.flags)
print(images.nbytes)
traffic_sign = [300, 2250, 3650, 4000]
# for i in range(len(traffic_sign)):
#     plt.subplot(1, 4, i+1)
#     plt.axis("off")
#     plt.imshow(images[traffic_sign[i]])
#     plt.subplots_adjust(wspace=0.5)
#     plt.show()
#     print("shape: {0}, min: {1}, max: {2}".format(images[traffic_sign[i]].shape,images[traffic_sign[i]].min(),images[traffic_sign[i]].max()))
# plt.hist(labels, 62)
# plt.show()

images28 = [transform.resize(image, (28, 28)) for image in images]
images28 = np.array(images28, dtype=np.float32)
images28 = rgb2gray(images28)
unique_labels = set(labels)
i = 1
show_images = False
if show_images:
    plt.figure(figsize=(15,15))
    for label in unique_labels:
        image = images28[labels.index(label)]
        plt.subplot(8,8,i)
        plt.axis("off")
        plt.title("Label :{0}, ({1})".format(label, labels.count(label)))
        i += 1
        plt.imshow(image, cmap="gray")
    plt.show()
labels= np.array(labels, dtype = np.int32)
print(images28.shape)
print(labels.dtype)
print(images28.dtype)
x = tf.placeholder(dtype = tf.float32, shape =[None, 28, 28])
y = tf.placeholder(dtype = tf.int32, shape =[None])
images_flat = tf.contrib.layers.flatten(x)

logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))

train_op = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

correct_pred = tf.argmax(logits, 1)

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


tf.set_random_seed(1234)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(201):
    print('EPOCH', i)
    _, accuracy_val = sess.run([train_op, accuracy], feed_dict = {x: images28, y:labels})
    if i % 10 == 0:
        print("Loss: ", loss)
    print('DONE WITH EPOCH')
sample_indexes = random.sample(range(len(images28)), 10)
sample_images = [images28[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

# Run the "correct_pred" operation
predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]
                        
# Print the real and predicted labels
print(sample_labels)
print(predicted)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
             fontsize=12, color=color)
    plt.imshow(sample_images[i],  cmap="gray")

plt.show()



