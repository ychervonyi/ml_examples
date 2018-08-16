# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

  # Parameters
  learning_rate = 0.01
  training_epochs = 10
  batch_size = 128
  display_step = 1

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  pred = tf.nn.softmax(tf.matmul(x, W) + b)

  # Define loss and optimizer
  y = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  # cross_entropy = tf.reduce_mean(
  #     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

  # Initializing the variables
  init = tf.global_variables_initializer()

  cost = cross_entropy
  # Test model
  correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
  # Calculate accuracy
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

  # Launch the graph
  with tf.Session() as sess:
      sess.run(init)

      # Training cycle
      for epoch in range(training_epochs):
          avg_cost = 0.
          total_batch = int(mnist.train.num_examples / batch_size)
          # Loop over all batches
          for i in range(total_batch):
              batch_x, batch_y = mnist.train.next_batch(batch_size)
              # Run optimization op (backprop) and cost op (to get loss value)
              _, c, acc = sess.run([optimizer, cost, accuracy], feed_dict={x: batch_x,
                                                                           y: batch_y})
              # Compute average loss
              avg_cost += c / total_batch
          # Display logs per epoch step
          if epoch % display_step == 0:
              print('Accuracy at epoch %s: train %s test %s' % (
              epoch, acc, accuracy.eval({x: mnist.test.images, y: mnist.test.labels})))
      print("Optimization Finished!")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)