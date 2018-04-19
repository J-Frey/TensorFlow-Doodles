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

	# Model - Softmax Convolutional NN

	# Build First Convolutional Layer

	# Convolution with 5x5 patch, 1 input channel, and 32 feature output channel
	W_conv1 = create_weight_variable([5, 5, 1, 32])
	b_conv1 = create_bias_variable([32])

	# Reshape x to a 4d tensor
	# shape = [?, width, height, #color channels]
	x_image = tf.reshape(x, [-1, 28, 28, 1])

	# Convolve x_image with weight tensor, add bias, apply ReLU, max pool
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	# Build Second Convolutional Layer

	# 5x5 path, 32 input chanel, 64 output channel
	W_conv2 = create_weight_variable([5, 5, 32, 64])
	b_conv2 = create_bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	# Densely Connected Layer

	# Currently image reduced to 7x7
	# Add fully-connected layer with 2014 neurons to process entire image
	# Reshape tensor from pooling layer into batch of vectos, multiply by weight, add bias, ReLU
	W_fc1 = create_weight_variable([7 * 7 * 64, 1024])
	b_fc1 = create_bias_variable([1024])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	# Dropout - Reduce Overfit

	# Allows toggle for dropout (on for training, off for testing)
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	# Readout Layer
	W_fc2 = create_weight_variable([1024, nClass])
	b_fc2 = create_bias_variable([nClass])

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


	# Loss Function based on Cross-Entropy
	y_ = tf.placeholder(tf.float32, [None, nClass])
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
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
		for i in range(5000):
			# Get a small subset batch of samples from training set (Stochastic Training)
			# Keep batch size low. Larger sets are computationally expensive
			batch_xs, batch_ys = mnist.train.next_batch(50)

			# Log Current Accuracy every 100 Iterations
			if i % 100 == 0:
				train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
				print("\tStep %d: training accuracy %g" % (i, train_accuracy))

			# Train Model
			train_step.run(feed_dict={x: batch_xs, y_:batch_ys, keep_prob: 0.5})

		# Report Model Accuracy for test set
		print("Test Accuracy %g" % accuracy.eval(feed_dict={
			x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", 
			type=str, 
			default="/tmp/tensorflow/mnist/input_data", 
			help="Directory for storing input data")
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)