import tensorflow as tf
import numpy as np
import pickle
import random


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
W_Conv1 = weight_variable([5, 5, 3, 32])
b_Conv1 = bias_variable([32])

h_Conv1 = tf.nn.relu(conv2d(inputReshaped, W_Conv1) + b_Conv1)
h_pool1 = max_pool_2x2(h_Conv1)

#Convolution2
W_Conv2 = weight_variable([5, 5, 32, 48])
b_Conv2 = bias_variable([48])

h_Conv2 = tf.nn.relu(conv2d(h_pool1, W_Conv2) + b_Conv2)
h_pool2 = max_pool_2x2(h_Conv2)

W_fc1 = weight_variable([8 * 8 * 48, 1624])
b_fc1 = bias_variable([1624])

h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*48])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Readout layer
W_fc2 = weight_variable([1624, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



image_batch = []
label_batch = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    i = 1
    for i in range(5):
        image_batch.clear()
        label_batch.clear()
        for j in range(500):
            l = random.randint(0, 49999)
            image_batch.append(cifarTrainData['data'][l])
            label_batch.append(cifarTrainData['labels'][l])
            if i % 2 == 0:
                train_accuracy = accuracy.eval(feed_dict={neuralinput: cifarTestData['data'], labels: cifarTestData['labels'], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={neuralinput: image_batch, labels: label_batch, keep_prob: 0.5})

        print('test accuracy %g' % accuracy.eval(feed_dict={
            neuralinput: cifarTestData['data'], labels: cifarTestData['labels'], keep_prob: 1.0}))













