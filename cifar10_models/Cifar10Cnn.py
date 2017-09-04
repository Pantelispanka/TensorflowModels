import tensorflow as tf
import numpy as np
import pickle
import random
# import cifar10_models.Cifar10Data as CifData
from Cifar10Data import Cifar10Dataset

cifar10Data = Cifar10Dataset("/home/pantelispanka/Python/TensorflowModels/cifar10_models/cifar-10-batches-py/")


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


#Input
neuralinput = tf.placeholder(tf.float32, [None, 3072], name='neuralinput')
labels = tf.placeholder(tf.float32, [None, 10], name='labels')
inputReshaped = tf.reshape(neuralinput, [-1, 32, 32, 3])

#Convolution1
W_Conv1 = weight_variable([5, 5, 3, 12])
b_Conv1 = bias_variable([12])

h_Conv1 = tf.nn.relu(conv2d(inputReshaped, W_Conv1) + b_Conv1)
h_pool1 = max_pool_2x2(h_Conv1)

#Convolution2
W_Conv2 = weight_variable([5, 5, 12, 18])
b_Conv2 = bias_variable([18])

h_Conv2 = tf.nn.relu(conv2d(h_pool1, W_Conv2) + b_Conv2)
h_pool2 = max_pool_2x2(h_Conv2)

W_fc1 = weight_variable([8 * 8 * 18, 1624])
b_fc1 = bias_variable([1624])

h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*18])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Readout layer
W_fc2 = weight_variable([1624, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# train_step = tf.train.AdadeltaOptimizer(1e-4).minimize(cross_entropy)
# train_step = tf.train.GradientDescentOptimizer(0.004).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    max_steps = 25000
    for step in range(max_steps):
        train_data, train_labels = cifar10Data.train_data_batch(100)
        if (step % 100) == 0:
            print(step, sess.run(accuracy,
                                 feed_dict={neuralinput: cifar10Data.test_data(), labels: cifar10Data.test_data_labels(),
                                            keep_prob: 1.0}))
        sess.run(train_step, feed_dict={neuralinput: train_data, labels: train_labels, keep_prob: 0.5})
    print(max_steps, sess.run(accuracy, feed_dict={neuralinput: cifar10Data.test_data(),
                                                   labels: cifar10Data.test_data_labels(), keep_prob: 1.0}))


# image_batch = []
# label_batch = []
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     i = 1
#     for i in range(12):
#         image_batch.clear()
#         label_batch.clear()
#         for j in range(1000):
#             l = random.randint(0, 49999)
#             image_batch.append(cifarTrainData['data'][l])
#             label_batch.append(cifarTrainData['labels'][l])
#             if i % 2 == 0:
#                 train_accuracy = accuracy.eval(feed_dict={neuralinput: cifarTestData['data'], labels: cifarTestData['labels'], keep_prob: 1.0})
#                 print('step %d, training accuracy %g' % (i, train_accuracy))
#             train_step.run(feed_dict={neuralinput: image_batch, labels: label_batch, keep_prob: 0.5})
#
#         print('test accuracy %g' % accuracy.eval(feed_dict={
#             neuralinput: cifarTestData['data'], labels: cifarTestData['labels'], keep_prob: 1.0}))













