'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 128
display_step = 1

# Network Parameters
n_hidden_1 = 512 # 1st layer number of features
n_hidden_2 = 512 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])




# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
weights = {
    'h1': tf.get_variable("h1", shape=[n_input, n_hidden_1], initializer=tf.contrib.layers.xavier_initializer()),
    'h2': tf.get_variable("h2", shape=[n_hidden_1, n_hidden_2], initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable("out", shape=[n_hidden_2, n_classes], initializer=tf.contrib.layers.xavier_initializer())
}
biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
    'out': tf.Variable(tf.zeros([n_classes]))
}



def model(x, weights, biases, dropout, phase_train):
    # Create model
    import numpy as np
    seed = np.random.randint(10e6)
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.layers.batch_normalization(layer_1, training=phase_train, renorm=True)
    # layer_1 = tf.contrib.layers.batch_norm(layer_1, is_training=phase_train, updates_collections=tf.GraphKeys.UPDATE_OPS, scope='batch_norm_1')
    layer_1 = tf.nn.relu(layer_1)
    # layer_1 = tf.nn.dropout(layer_1 * 1., dropout, noise_shape=None, seed=seed)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.layers.batch_normalization(layer_2, training=phase_train, renorm=True)
    # layer_2 = tf.contrib.layers.batch_norm(layer_2, is_training=phase_train, updates_collections=tf.GraphKeys.UPDATE_OPS, scope='batch_norm_2')
    layer_2 = tf.nn.relu(layer_2)
    # layer_2 = tf.nn.dropout(layer_2 * 1., dropout, noise_shape=None, seed=seed)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    # out_layer = tf.Print(out_layer, [out_layer, biases['out']], summarize=10)
    return out_layer


def model(x, weights, biases, dropout, phase_train):
    # Create model
    import numpy as np
    seed = np.random.randint(10e6)
    # Hidden layer with RELU activation
    layer_1 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(x, weights['h1']), training=phase_train, renorm=True))
    layer_1 = tf.nn.dropout(layer_1 * 1., dropout, noise_shape=None, seed=seed)
    layer_2 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(layer_1, weights['h2']), training=phase_train, renorm=True))
    layer_2 = tf.nn.dropout(layer_2 * 1., dropout, noise_shape=None, seed=seed)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    # out_layer = tf.Print(out_layer, [out_layer, biases['out']], summarize=10)
    return out_layer


keep_prob = tf.placeholder(tf.float32)
phase_train = tf.placeholder(tf.bool)

pred = model(x, weights, biases, keep_prob, phase_train)
softmax = tf.nn.softmax(pred)
tf.summary.histogram('softmax', softmax)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)) # softmax insider does not matter
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# In order to use tf.layers.batch_normalization need the following
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('/tmp/tensorflow/mnist/logs/mnist_bn/train')
test_writer = tf.summary.FileWriter('/tmp/tensorflow/mnist/logs/mnist_bn/test')
# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            summary, _, c, acc, _ = sess.run([merged, optimizer, cost, accuracy, extra_update_ops], \
                                 feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5, phase_train: True})
            train_writer.add_summary(summary, i)
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            summary, _, c, acc_test, _ = sess.run([merged, optimizer, cost, accuracy, extra_update_ops], \
                                             feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0, phase_train: False})
            train_writer.add_summary(summary, i)
            print('Accuracy at epoch %s: train %s test %s' % \
                  (epoch, acc, accuracy.eval({x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0, phase_train: False})))
    print("Optimization Finished!")

