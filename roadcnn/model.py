import numpy
import tensorflow as tf
import os
import os.path
import random
import math
import time
from PIL import Image

BATCH_SIZE = 2
KERNEL_SIZE = 3

class Model:
	def _conv_layer(self, name, input_var, stride, in_channels, out_channels, options = {}):
		activation = options.get('activation', 'relu')
		dropout = options.get('dropout', None)
		padding = options.get('padding', 'SAME')
		batchnorm = options.get('batchnorm', True)
		transpose = options.get('transpose', False)

		with tf.variable_scope(name) as scope:
			if not transpose:
				filter_shape = [KERNEL_SIZE, KERNEL_SIZE, in_channels, out_channels]
			else:
				filter_shape = [KERNEL_SIZE, KERNEL_SIZE, out_channels, in_channels]
			kernel = tf.get_variable(
				'weights',
				shape=filter_shape,
				initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / KERNEL_SIZE / KERNEL_SIZE / in_channels)),
				dtype=tf.float32
			)
			biases = tf.get_variable(
				'biases',
				shape=[out_channels],
				initializer=tf.constant_initializer(0.0),
				dtype=tf.float32
			)
			if not transpose:
				output = tf.nn.bias_add(
					tf.nn.conv2d(
						input_var,
						kernel,
						[1, stride, stride, 1],
						padding=padding
					),
					biases
				)
			else:
				batch = tf.shape(input_var)[0]
				side = tf.shape(input_var)[1]
				output = tf.nn.bias_add(
					tf.nn.conv2d_transpose(
						input_var,
						kernel,
						[batch, side * stride, side * stride, out_channels],
						[1, stride, stride, 1],
						padding=padding
					),
					biases
				)
			if batchnorm:
				output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=self.is_training, decay=0.99)
			if dropout is not None:
				output = tf.nn.dropout(output, keep_prob=1-dropout)

			if activation == 'relu':
				return tf.nn.relu(output, name=scope.name)
			elif activation == 'sigmoid':
				return tf.nn.sigmoid(output, name=scope.name)
			elif activation == 'none':
				return output
			else:
				raise Exception('invalid activation {} specified'.format(activation))

	def __init__(self, big=False):
		tf.reset_default_graph()

		self.is_training = tf.placeholder(tf.bool)
		if big:
			self.inputs = tf.placeholder(tf.float32, [None, 2048, 2048, 3])
			self.targets = tf.placeholder(tf.float32, [None, 2048, 2048, 1])
		else:
			self.inputs = tf.placeholder(tf.float32, [None, 256, 256, 3])
			self.targets = tf.placeholder(tf.float32, [None, 256, 256, 1])
		self.learning_rate = tf.placeholder(tf.float32)

		self.dropout_factor = tf.to_float(self.is_training) * 0.3

		self.layer1 = self._conv_layer('layer1', self.inputs, 2, 3, 128, {'batchnorm': False}) # -> 128x128x128
		self.layer2 = self._conv_layer('layer2', self.layer1, 1, 128, 128) # -> 128x128x128
		self.layer3 = self._conv_layer('layer3', self.layer2, 2, 128, 128) # -> 64x64x128
		self.layer4 = self._conv_layer('layer4', self.layer3, 1, 128, 128) # -> 64x64x128
		self.layer5 = self._conv_layer('layer5', self.layer4, 2, 128, 256, {'dropout': self.dropout_factor}) # -> 32x32x256
		self.layer6 = self._conv_layer('layer6', self.layer5, 1, 256, 256, {'dropout': self.dropout_factor}) # -> 32x32x256
		self.layer7 = self._conv_layer('layer7', self.layer6, 2, 256, 512, {'dropout': self.dropout_factor}) # -> 16x16x512
		self.layer8 = self._conv_layer('layer8', self.layer7, 2, 512, 512, {'dropout': self.dropout_factor}) # -> 8x8x512
		self.layer9 = self._conv_layer('layer9', self.layer8, 1, 512, 512, {'dropout': self.dropout_factor}) # -> 8x8x512
		self.layer10 = self._conv_layer('layer10', self.layer9, 1, 512, 512, {'dropout': self.dropout_factor}) # -> 8x8x512
		self.layer11 = self._conv_layer('layer11', self.layer10, 1, 512, 512, {'dropout': self.dropout_factor}) # -> 8x8x512
		self.layer12 = self._conv_layer('layer12', self.layer11, 2, 512, 512, {'transpose': True, 'dropout': self.dropout_factor}) # -> 16x16x512
		self.layer13_inputs = tf.concat([self.layer7, self.layer12], axis=3)
		self.layer13 = self._conv_layer('layer13', self.layer13_inputs, 1, 1024, 512, {'dropout': self.dropout_factor}) # -> 16x16x512
		self.layer14 = self._conv_layer('layer14', self.layer13, 2, 512, 256, {'transpose': True, 'dropout': self.dropout_factor}) # -> 32x32x256
		self.layer15_inputs = tf.concat([self.layer6, self.layer14], axis=3)
		self.layer15 = self._conv_layer('layer15', self.layer15_inputs, 1, 512, 256, {'dropout': self.dropout_factor}) # -> 32x32x256
		self.layer16 = self._conv_layer('layer16', self.layer15, 2, 256, 128, {'transpose': True}) # -> 64x64x128
		self.layer17_inputs = tf.concat([self.layer4, self.layer16], axis=3)
		self.layer17 = self._conv_layer('layer17', self.layer17_inputs, 1, 256, 128) # -> 64x64x128
		self.layer18 = self._conv_layer('layer18', self.layer17, 2, 128, 128, {'transpose': True}) # -> 128x128x128
		self.layer19_inputs = tf.concat([self.layer2, self.layer18], axis=3)
		self.layer19 = self._conv_layer('layer19', self.layer19_inputs, 2, 256, 128, {'transpose': True}) # -> 256x256x128
		self.layer20 = self._conv_layer('layer20', self.layer19, 1, 128, 128) # -> 256x256x128
		self.pre_outputs = self._conv_layer('pre_outputs', self.layer20, 1, 128, 2, {'activation': 'none', 'batchnorm': False}) # -> 256x256x2

		self.outputs = tf.nn.softmax(self.pre_outputs)[:, :, :, 0]
		self.labels = tf.concat([self.targets, 1 - self.targets], axis=3)
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.pre_outputs))

		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

		self.init_op = tf.initialize_all_variables()
		self.saver = tf.train.Saver(max_to_keep=None)
