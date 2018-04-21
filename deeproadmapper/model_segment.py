import numpy
import tensorflow as tf
import os
import os.path
import random
import math
import time
from PIL import Image

BATCH_SIZE = 1
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

	def _residual_layer(self, name, input_var, stride, in_channels, out_channels, options = {}):
		def bn_relu_conv_layer(name, input_var, stride, in_channels, out_channels, options = {}):
			with tf.variable_scope(name):
				padding = options.get('padding', 'SAME')
				transpose = options.get('transpose', False)
				first_block = options.get('first_block', False)

				if not first_block:
					input_var = tf.contrib.layers.batch_norm(input_var, center=True, scale=True, is_training=self.is_training, decay=0.99)
					input_var = tf.nn.relu(input_var)

				batch = tf.shape(input_var)[0]
				side = tf.shape(input_var)[1]

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
					conv_output = tf.nn.bias_add(
						tf.nn.conv2d(
							input_var,
							kernel,
							[1, stride, stride, 1],
							padding=padding
						),
						biases
					)
				else:
					conv_output = tf.nn.bias_add(
						tf.nn.conv2d_transpose(
							input_var,
							kernel,
							[batch, side * stride, side * stride, out_channels],
							[1, stride, stride, 1],
							padding=padding
						),
						biases
					)

				return conv_output


		padding = options.get('padding', 'SAME')
		transpose = options.get('transpose', False)
		first_block = options.get('first_block', False)

		with tf.variable_scope(name):
			conv1 = bn_relu_conv_layer('conv1', input_var, stride, in_channels, out_channels, {'padding': padding, 'transpose': transpose, 'first_block': first_block})
			conv2 = bn_relu_conv_layer('conv2', input_var, 1, out_channels, out_channels, {'padding': padding})

			if stride == 1:
				output = input_var + conv2
			elif not transpose:
				fix_input = tf.nn.avg_pool(input_var, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
				if out_channels > in_channels:
					fix_input = tf.pad(fix_input, [[0, 0], [0, 0], [0, 0], [in_channels // 2, in_channels // 2]])
				output = fix_input + conv2
			else:
				fix_input = tf.image.resize_images(input_var, [side * stride, side * stride], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
				output = fix_input + conv2

		return output

	def _fc_layer(self, name, input_var, input_size, output_size, options = {}):
		activation = options.get('activation', 'relu')
		dropout = options.get('dropout', None)
		batchnorm = options.get('batchnorm', True)

		with tf.variable_scope(name) as scope:
			weights = tf.get_variable(
				'weights',
				shape=[input_size, output_size],
				initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / input_size)),
				dtype=tf.float32
			)
			biases = tf.get_variable(
				'biases',
				shape=[output_size],
				initializer=tf.constant_initializer(0.0),
				dtype=tf.float32
			)
			output = tf.matmul(input_var, weights) + biases
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

	def __init__(self, mode=0, big=False):
		tf.reset_default_graph()

		self.is_training = tf.placeholder(tf.bool)
		if big:
			self.inputs = tf.placeholder(tf.float32, [None, 4096, 4096, 3])
			self.targets = tf.placeholder(tf.float32, [None, 4096, 4096, 1])
		else:
			self.inputs = tf.placeholder(tf.float32, [None, 256, 256, 3])
			self.targets = tf.placeholder(tf.float32, [None, 256, 256, 1])
		self.learning_rate = tf.placeholder(tf.float32)

		self.layer1 = self._conv_layer('layer1', self.inputs, 1, 3, 16, {'batchnorm': False}) # -> 1024x1024
		self.layer2 = self._residual_layer('layer2', self.layer1, 1, 16, 16, {'first_block': True}) # -> 1024x1024
		self.layer3 = self._residual_layer('layer3', self.layer2, 1, 16, 16) # -> 1024x1024
		self.layer4 = self._residual_layer('layer4', self.layer3, 1, 16, 16) # -> 1024x1024
		self.layer5 = self._residual_layer('layer5', self.layer4, 1, 16, 16) # -> 1024x1024
		self.layer6 = self._residual_layer('layer6', self.layer5, 1, 16, 16) # -> 1024x1024
		self.layer7 = self._residual_layer('layer7', self.layer6, 1, 16, 16) # -> 1024x1024

		self.layer8 = self._conv_layer('layer8', self.layer7, 2, 16, 32) # -> 512x512
		self.layer9 = self._residual_layer('layer9', self.layer8, 1, 32, 32) # -> 512x512
		self.layer10 = self._residual_layer('layer10', self.layer9, 1, 32, 32) # -> 512x512
		self.layer11 = self._residual_layer('layer11', self.layer10, 1, 32, 32) # -> 512x512
		self.layer12 = self._residual_layer('layer12', self.layer11, 1, 32, 32) # -> 512x512
		self.layer13 = self._residual_layer('layer13', self.layer12, 1, 32, 32) # -> 512x512
		self.layer14 = self._residual_layer('layer14', self.layer13, 1, 32, 32) # -> 512x512

		self.layer15 = self._conv_layer('layer15', self.layer14, 2, 32, 64) # -> 256x256
		self.layer16 = self._residual_layer('layer16', self.layer15, 1, 64, 64) # -> 256x256
		self.layer17 = self._residual_layer('layer17', self.layer16, 1, 64, 64) # -> 256x256
		self.layer18 = self._residual_layer('layer18', self.layer17, 1, 64, 64) # -> 256x256
		self.layer19 = self._residual_layer('layer19', self.layer18, 1, 64, 64) # -> 256x256
		self.layer20 = self._residual_layer('layer20', self.layer19, 1, 64, 64) # -> 256x256
		self.layer21 = self._residual_layer('layer21', self.layer20, 1, 64, 64) # -> 256x256

		self.layer22 = self._conv_layer('layer22', self.layer21, 2, 64, 128) # -> 128x128
		self.layer23 = self._residual_layer('layer23', self.layer22, 1, 128, 128) # -> 128x128
		self.layer24 = self._residual_layer('layer24', self.layer23, 1, 128, 128) # -> 128x128
		self.layer25 = self._residual_layer('layer25', self.layer24, 1, 128, 128) # -> 128x128
		self.layer26 = self._residual_layer('layer26', self.layer25, 1, 128, 128) # -> 128x128
		self.layer27 = self._residual_layer('layer27', self.layer26, 1, 128, 128) # -> 128x128
		self.layer28 = self._residual_layer('layer28', self.layer27, 1, 128, 128) # -> 128x128

		self.decoder1_inputs = tf.concat([self.layer28, self.layer22], axis=3)
		self.decoder1 = self._conv_layer('decoder1', self.decoder1_inputs, 2, 2*128, 64, {'transpose': True}) # -> 256x256
		self.decoder2_inputs = tf.concat([self.decoder1, self.layer15, self.layer21], axis=3)
		self.decoder2 = self._conv_layer('decoder2', self.decoder2_inputs, 2, 3*64, 32, {'transpose': True}) # -> 512x512
		self.decoder3_inputs = tf.concat([self.decoder2, self.layer8, self.layer14], axis=3)
		self.decoder3 = self._conv_layer('decoder3', self.decoder3_inputs, 2, 3*32, 16, {'transpose': True}) # -> 1024x1024

		self.initial_outputs = self._conv_layer('initial_outputs', self.decoder3, 1, 16, 2, {'activation': 'none', 'batchnorm': False}) # -> 1024x1024
		self.pre_outputs = tf.nn.softmax(self.initial_outputs)
		self.outputs = self.pre_outputs[:, :, :, 0]

		EPSILON = 1.0
		self.loss_numerator1 = tf.reduce_sum(self.pre_outputs[:, :, :, 0:1] * self.targets) + EPSILON
		self.loss_denominator1 = tf.reduce_sum(self.pre_outputs[:, :, :, 0:1] + self.targets - self.pre_outputs[:, :, :, 0:1] * self.targets) + EPSILON
		self.loss_numerator2 = tf.reduce_sum(self.pre_outputs[:, :, :, 1:2] * (1 - self.targets)) + EPSILON
		self.loss_denominator2 = tf.reduce_sum(self.pre_outputs[:, :, :, 1:2] + (1 - self.targets) - self.pre_outputs[:, :, :, 1:2] * (1 - self.targets)) + EPSILON
		if mode == 0:
			self.loss = -(self.loss_numerator1 / self.loss_denominator1 + self.loss_numerator2 / self.loss_denominator2 / 13)
		elif mode == 1:
			self.loss = -(self.loss_numerator1 / self.loss_denominator1 + self.loss_numerator2 / self.loss_denominator2)
		elif mode == 2:
			self.loss = -(self.loss_numerator1 / self.loss_denominator1)
		elif mode == 3:
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.concat([self.targets, 1 - self.targets], axis=3), logits=self.initial_outputs))
		elif mode == 4:
			self.loss = -(self.loss_numerator1 / self.loss_denominator1 + self.loss_numerator2 / self.loss_denominator2 / 2)

		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

		self.init_op = tf.initialize_all_variables()
		self.saver = tf.train.Saver(max_to_keep=None)
