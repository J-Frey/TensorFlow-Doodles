## Original Project: https://www.tensorflow.org/get_started/mnist/beginners

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def create_weight_variable(shape):
	""" Creates a weight variable.
	Initialized with 0.1 to avoid zero gradient
	"""

	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def create_bias_variable(shape):
	""" Creates a bias variable. 
	Initialized with 0.1 to avoid dead neurons.
	"""
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



def main(_):
	# Load input data
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
	#print(mnist[0])
	#exit()
	# Number of dimensions of vector
	nDim = 784

	# Number of classes
	nClass = 10

	# Input
	x = tf.placeholder(tf.float32, [None, nDim])

	# Model - Softmax Regression
	# Weights - To be trained
	W = tf.Variable(tf.truncated_normal([nDim, nClass], stddev=0.1))

	# Bias - To be trained
	b = tf.Variable(tf.constant(0.1, shape=[nClass]))

	# Model - Softmax Regression
	# Note: x and W are flipped "as a small trick to deal with x being a 2D tensor with multiple inputs."
	y = tf.nn.softmax(tf.matmul(x, W) + b)

	# Loss Function based on Cross-Entropy
	y_ = tf.placeholder(tf.float32, [None, nClass])

	with tf.name_scope('loss'):
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

	# Explicit version, however, is numerically unstable. Do not use
	#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

	# Gradient Optimizer
	# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	
	# Adam Optimizer 
	with tf.name_scope('adam_optimizer'):
		train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	
	# Accuracy Variables
	with tf.name_scope('accuracy'):
		correct_prediction = tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


	graph_location = tempfile.mkdtemp()
	print("Saving graph to: %s" % graph_location)
	train_writer = tf.summary.FileWriter(graph_location)
	train_writer.add_graph(tf.get_default_graph())


	# Create session
	with tf.Session() as sess:
		# Init variables
		sess.run(tf.global_variables_initializer())

		# Train Model
		for i in range(10000):
			# Get a small subset batch of samples from training set (Stochastic Training)
			# Keep batch size low. Larger sets are computationally expensive
			batch_xs, batch_ys = mnist.train.next_batch(1000)

			# Log Current Accuracy every 100 Iterations
			if i % 100 == 0:
				train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_:batch_ys})
				print("\tStep %d: training accuracy %g" % (i, train_accuracy))
			
			# Train Model
			train_step.run(feed_dict={x: batch_xs, y_: batch_ys})

		# Report Model Accuracy for test set
		print("Test Accuracy %g" % accuracy.eval(feed_dict={
			x: mnist.test.images, y_: mnist.test.labels}))



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", 
			type=str, 
			default="/tmp/tensorflow/mnist/input_data", 
			help="Directory for storing input data")
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)