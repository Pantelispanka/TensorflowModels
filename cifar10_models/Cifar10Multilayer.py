import tensorflow as tf
import numpy as np
import pickle
from tensorflow.examples.tutorials.mnist import input_data
import random

import math

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        return dict


X1 = unpickle('/Users/pantelispanka/Python/pythonNotebook/data/cifar-10-batches-py/data_batch_1')
X2 = unpickle('/Users/pantelispanka/Python/pythonNotebook/data/cifar-10-batches-py/data_batch_2')
X3 = unpickle('/Users/pantelispanka/Python/pythonNotebook/data/cifar-10-batches-py/data_batch_3')
X4 = unpickle('/Users/pantelispanka/Python/pythonNotebook/data/cifar-10-batches-py/data_batch_4')
X5 = unpickle('/Users/pantelispanka/Python/pythonNotebook/data/cifar-10-batches-py/data_batch_5')
X6 = unpickle('/Users/pantelispanka/Python/pythonNotebook/data/cifar-10-batches-py/test_batch')

cifarTestDataMatrix = X6[b'data']
cifarTestDataLabelsToVec = X6[b'labels']

cifarTrainDataMatrix = np.concatenate((X1[b'data'], X2[b'data'], X3[b'data'], X4[b'data'], X5[b'data']))
cifarTrainDataLabelsToVec = np.concatenate((X1[b'labels'], X2[b'labels'], X3[b'labels'], X4[b'labels'], X5[b'labels']))


cifarTestDataLabelsList = []

for i in range(10000):
    if cifarTestDataLabelsToVec[i] == 1:
        cifarTestDataLabelsList.append([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
    elif cifarTestDataLabelsToVec[i] == 2:
        cifarTestDataLabelsList.append([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.])
    elif cifarTestDataLabelsToVec[i] == 3:
        cifarTestDataLabelsList.append([0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
    elif cifarTestDataLabelsToVec[i] == 4:
        cifarTestDataLabelsList.append([0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])
    elif cifarTestDataLabelsToVec[i] == 5:
        cifarTestDataLabelsList.append([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.])
    elif cifarTestDataLabelsToVec[i] == 6:
        cifarTestDataLabelsList.append([0., 0., 0., 0., 0., 0., 1., 0., 0., 0.])
    elif cifarTestDataLabelsToVec[i] == 7:
        cifarTestDataLabelsList.append([0., 0., 0., 0., 0., 0., 0., 1., 0., 0.])
    elif cifarTestDataLabelsToVec[i] == 8:
        cifarTestDataLabelsList.append([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.])
    elif cifarTestDataLabelsToVec[i] == 9:
        cifarTestDataLabelsList.append([0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])
    elif cifarTestDataLabelsToVec[i] == 0:
        cifarTestDataLabelsList.append([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])


cifarTrainDataLabelsList = []

for i in range(50000):
    if cifarTrainDataLabelsToVec[i] == 1:
        cifarTrainDataLabelsList.append([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
    elif cifarTrainDataLabelsToVec[i] == 2:
        cifarTrainDataLabelsList.append([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.])
    elif cifarTrainDataLabelsToVec[i] == 3:
        cifarTrainDataLabelsList.append([0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
    elif cifarTrainDataLabelsToVec[i] == 4:
        cifarTrainDataLabelsList.append([0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])
    elif cifarTrainDataLabelsToVec[i] == 5:
        cifarTrainDataLabelsList.append([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.])
    elif cifarTrainDataLabelsToVec[i] == 6:
        cifarTrainDataLabelsList.append([0., 0., 0., 0., 0., 0., 1., 0., 0., 0.])
    elif cifarTrainDataLabelsToVec[i] == 7:
        cifarTrainDataLabelsList.append([0., 0., 0., 0., 0., 0., 0., 1., 0., 0.])
    elif cifarTrainDataLabelsToVec[i] == 8:
        cifarTrainDataLabelsList.append([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.])
    elif cifarTrainDataLabelsToVec[i] == 9:
        cifarTrainDataLabelsList.append([0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])
    elif cifarTrainDataLabelsToVec[i] == 0:
        cifarTrainDataLabelsList.append([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])


cifarTrainData = {'data': ((cifarTrainDataMatrix / 255) - np.mean(cifarTrainDataMatrix / 255)), 'labels': np.asarray(cifarTrainDataLabelsList)}


cifarTestData = {'data': ((cifarTestDataMatrix / 255) - np.mean(cifarTestDataMatrix / 255)), 'labels': np.asarray(cifarTestDataLabelsList)}


#Parameters
n_hidden1 = 2200
n_hidden2 = 800
n_input = 784 #3072


neulalinput = tf.placeholder(tf.float32, [None, n_input])

labels = tf.placeholder(tf.float32, [None, 10])


def multilayer_perceptron(x, weights, biases):
    layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer1 = tf.nn.sigmoid(layer1)
    layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])
    layer2 = tf.nn.sigmoid(layer2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer2, weights['out']) + biases['out']
    return out_layer

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden1])),
    'h2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),
    'out': tf.Variable(tf.random_normal([n_hidden2, 10]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden1])),
    'b2': tf.Variable(tf.random_normal([n_hidden2])),
    'out': tf.Variable(tf.random_normal([10]))
}

model = multilayer_perceptron(neulalinput, weights, biases)

cost = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(model), reduction_indices=[1]))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=labels))
optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(cost)


init = tf.global_variables_initializer()

# with tf.Session() as sess:
#     sess.run(init)
#
#     image_batch = []
#     label_batch = []
#
#     for i in range(100):
#         # image_batch = image_batch.append(cifarTrainData['data'][i])
#         # label_batch = label_batch.append(cifarTrainData['labels'][i])
#         image_batch.clear()
#         label_batch.clear()
#         for j in range(228):
#             l = random.randint(0, 49999)
#             image_batch.append(cifarTrainData['data'][l])
#             label_batch.append(cifarTrainData['labels'][l])
#
#         print(sess.run([optimizer, cost], feed_dict={neulalinput: image_batch, labels: label_batch}))
#
#     correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(labels, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     print(sess.run([accuracy, cost], feed_dict={neulalinput: cifarTestData['data'], labels: cifarTestData['labels']}))


sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for _ in range(1000):
  # writer = tf.summary.FileWriter('./graphs', sess.graph)
  batch_xs, batch_ys = mnist.train.next_batch(100)
  print(sess.run([optimizer, cost], feed_dict={neulalinput: batch_xs, labels: batch_ys}))


correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
batch_xs, batch_ys = mnist.train.next_batch(1000)
print(sess.run(accuracy, feed_dict={neulalinput: batch_xs, labels: batch_ys}))
print(sess.run(accuracy, feed_dict={neulalinput: mnist.test.images, labels: mnist.test.labels}))