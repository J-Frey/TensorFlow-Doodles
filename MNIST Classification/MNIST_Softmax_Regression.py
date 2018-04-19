## Original Project: https://www.tensorflow.org/get_started/mnist/beginners

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

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

	# Weights - To be trained
	W = tf.Variable(tf.zeros([nDim, nClass]))

	# Bias - To be trained
	b = tf.Variable(tf.zeros([nClass]))

	# Model - Softmax Regression
	# Note: x and W are flipped "as a small trick to deal with x being a 2D tensor with multiple inputs."
	y = tf.nn.softmax(tf.matmul(x, W) + b)



	# Loss Function based on Cross-Entropy
	y_ = tf.placeholder(tf.float32, [None, nClass])
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

	# Explicit version, however, is numerically unstable. Do not use
	#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


	# Create session
	sess = tf.InteractiveSession()

	# Init variables
	tf.global_variables_initializer().run()

	# Train Model
	for _ in range(5000):
		# Get a subset batch of 100 samples from training set (Stochastic Training)
		# Keep batch size low. Larger sets are computationally expensive
		batch_xs, batch_ys = mnist.train.next_batch(100)

		# Train using this batch
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	# Testing trained model
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", 
			type=str, 
			default="/tmp/tensorflow/mnist/input_data", 
			help="Directory for storing input data")
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)