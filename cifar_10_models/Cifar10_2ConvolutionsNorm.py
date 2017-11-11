import tensorflow as tf
import numpy as np
# from cifar_10_models.Cifar10Data import Cifar10Dataset
from Cifar10Data import Cifar10Dataset

cifar10Data = Cifar10Dataset("/home/pantelispanka/Python/TensorflowModels/cifar_10_models/cifar-10-batches-py/")

# Input layer
input_layer = tf.placeholder(tf.float32, shape=[None, 3072], name='neural_input')

input_reshaped = tf.reshape(input_layer, shape=[-1, 32, 32, 3])

# First convolution
first_filter = tf.get_variable('filter_1', [5, 5, 3, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
first_convolution = tf.nn.conv2d(input_reshaped, filter=first_filter, strides=[1, 1, 1, 1], padding='SAME')
first_conv_bias = tf.Variable(tf.constant(0.1, shape=[64]))
first_conv_activation = tf.nn.relu(first_convolution + first_conv_bias)
first_conv_norm2 = tf.nn.lrn(first_conv_activation, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                              name='norm2')
first_pool = tf.nn.max_pool(first_conv_norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Second convolution
second_filter = tf.get_variable('filter_2', [5, 5, 64, 128], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
second_convolution = tf.nn.conv2d(input=first_pool, filter=second_filter, strides=[1, 1, 1, 1], padding='SAME')
second_conv_bias = tf.Variable(tf.constant(0.1, shape=[128]))
second_conv_activation = tf.nn.relu(second_convolution + second_conv_bias)
second_conv_norm2 = tf.nn.lrn(second_conv_activation, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                              name='norm2')
second_pool = tf.nn.max_pool(second_conv_norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Fully connected first
pool_flat = tf.reshape(second_pool, [-1, 8 * 8 * 128])
full_connected_weights = tf.Variable(tf.truncated_normal([8 * 8 * 128, 4028], stddev=0.1))
full_connected_bias = tf.Variable(tf.constant(0.1, shape=[4028]))

full_connected_activation = tf.nn.relu(tf.matmul(pool_flat, full_connected_weights) + full_connected_bias)


# Fully connected second
full_connected_2_weights = tf.Variable(tf.truncated_normal([4028, 1024], stddev=0.1))
full_connected_2_bias = tf.Variable(tf.constant(0.1, shape=[1024]))

full_connected_2_activation = tf.nn.relu(tf.matmul(full_connected_activation
                                                   , full_connected_2_weights) + full_connected_2_bias)


# Dropout
keep_prob = tf.placeholder(tf.float32)
drop = tf.nn.dropout(full_connected_2_activation, keep_prob)

# Readout layer
read_out_weights = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
read_out_bias = tf.Variable(tf.constant(0.1, shape=[10]))

neural_output = tf.matmul(drop, read_out_weights) + read_out_bias

labeled = tf.placeholder(tf.float32, shape=[None, 10], name="Labeled")

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labeled, logits=neural_output))
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(neural_output, 1), tf.argmax(labeled, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")


saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    writer.close()
    for i in range(1000):
        train_data = []
        train_labels = []
        train_data, train_labels = cifar10Data.train_data_batch(64)
        if (i % 100) == 0:
            train_data_eval, train_labels_eval = cifar10Data.train_data_for_eval()
            print("For step", i, "accuracy on train data is", sess.run(accuracy, feed_dict={
                input_layer: train_data_eval, labeled: train_labels_eval, keep_prob: 1.0
            }), "for the test data is:", sess.run(accuracy, feed_dict={
                    input_layer: cifar10Data.test_data(), labeled: cifar10Data.test_data_labels(), keep_prob: 1.0
                }))
        sess.run(train_step, feed_dict={
                    input_layer: train_data, labeled: train_labels, keep_prob: 0.5
                })
    writer.close()
    save_path = saver.save(sess, "./cifar102ConvNorm.ckpt")
    print("Model saved in file: %s" % save_path)
    print("Overall accuracy on test data is", sess.run(accuracy, feed_dict={
                    input_layer: cifar10Data.test_data(), labeled: cifar10Data.test_data_labels(), keep_prob: 1.0
                }))
