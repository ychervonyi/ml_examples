'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 128
display_step = 50
display_step = 1

# Network Parameters
n_hidden_1 = 512 # 1st layer number of features
n_hidden_2 = 512 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=1.0)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=1.0)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=1.0))
}
# default stddev is 1.0

biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
    'out': tf.Variable(tf.zeros([n_classes]))
}

variable_summaries(weights['h1'])
variable_summaries(weights['h2'])
variable_summaries(weights['out'])
variable_summaries(biases['b1'])
variable_summaries(biases['b2'])
variable_summaries(biases['out'])



def model(x, weights, biases, dropout):
    # Create model
    import numpy as np
    seed = np.random.randint(10e6)
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    tf.summary.histogram('layer_1', layer_1)
    layer_1 = tf.nn.relu(layer_1)
    tf.summary.histogram('layer_1_relu', layer_1)
    layer_1 = tf.nn.dropout(layer_1, dropout, noise_shape=None, seed=seed)
    tf.summary.histogram('layer_1_relu_dropout', layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    tf.summary.histogram('layer_2', layer_2)
    layer_2 = tf.nn.relu(layer_2)
    tf.summary.histogram('layer_2_relu', layer_2)
    layer_2 = tf.nn.dropout(layer_2, dropout, noise_shape=None, seed=seed)
    tf.summary.histogram('layer_2_relu_dropout', layer_2)
    # Output layer with linear activation
    # layer_2 = layer_1
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    tf.summary.histogram('out_layer', out_layer)
    # out_layer = tf.Print(out_layer, [out_layer, biases['out']], summarize=10)
    return out_layer

keep_prob = tf.placeholder(tf.float32)

pred = model(x, weights, biases, keep_prob)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
tf.summary.scalar('cross_entropy', cost)
# Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)) # softmax insider does not matter
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('tmp/bug/train')
test_writer = tf.summary.FileWriter('tmp/bug/test')

optimizers = [tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost),
              tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost),
              tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)]
optimizers_labels = ['Adam', 'GradientDecent', 'RMSProp']
# optimizers = [optimizers[2]]
# it seems like RMSProp gets out of a bad minimum after 20 epochs
accs_lists = []
for index, optimizer in enumerate(optimizers):
    # Initializing the variables
    init = tf.global_variables_initializer()
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        accs_train, accs_test = [], []
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                summary, _, c, acc = sess.run([merged, optimizer, cost, accuracy], \
                                     feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
                # Compute average loss
                avg_cost += c / total_batch
                # if i % display_step == 0:
                #     summary_test, _, _, acc_test = sess.run([merged, optimizer, cost, accuracy], \
                #                                   feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
                #     train_writer.add_summary(summary, i)
                #     test_writer.add_summary(summary_test, i)
                #     print('Accuracy at epoch %s: train %s test %s' % (epoch, acc, acc_test))
            # Display logs per epoch step
            if epoch % display_step == 0:
                summary_test, _, _, acc_test = sess.run([merged, optimizer, cost, accuracy], \
                                              feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
                accs_train.append(acc)
                accs_test.append(acc_test)
                train_writer.add_summary(summary, epoch)
                test_writer.add_summary(summary_test, epoch)
                print('Accuracy at epoch %s: train %s test %s' % (epoch, acc, acc_test))
        print("Optimization Finished!")
        accs_train = np.transpose(np.array(accs_train)).tolist()
        accs_test = np.transpose(np.array(accs_test)).tolist()
        accs_lists.append([accs_train, accs_test])
train_writer.close()
test_writer.close()
epochs = [i for i in range(training_epochs)]
fig = plt.figure(figsize=(15,10))
axes = plt.gca()
axes.set_ylim([0.0, 1.0])  # fix the scale
ax1 = fig.add_subplot(111)
for i in range(len(optimizers)):
    # plt.plot(epochs, accs_lists[i][0], label='Train '+optimizers_labels[i])
    plt.plot(epochs, accs_lists[i][1], label='Test '+optimizers_labels[i])
plt.legend()
plt.show()


