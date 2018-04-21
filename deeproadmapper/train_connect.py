import model_connect as model

import numpy
import os
from PIL import Image
import random
import scipy.ndimage
import subprocess
import sys
import tensorflow as tf
import time

data_path = sys.argv[1]
model_path = sys.argv[2]

print 'loading tiles'

def load_ims(path, limit=None):
	fnames = [path + '/' + fname for fname in os.listdir(path) if '.sat.png' in fname]
	if limit is not None and len(fnames) > limit:
		fnames = random.sample(fnames, limit)
	ims = []
	for sat_fname in fnames:
		con_fname = sat_fname.replace('.sat.', '.con.')
		im = numpy.zeros((320, 320, 6), dtype='uint8')
		im[:, :, 0:3] = scipy.ndimage.imread(sat_fname).swapaxes(0, 1)
		im[:, :, 3:6] = scipy.ndimage.imread(con_fname).swapaxes(0, 1)
		ims.append(im)
	return ims

def extract_example(tile):
	im, label = tile
	i = random.randint(0, 64)
	j = random.randint(0, 64)
	return im[i:i+256, j:j+256, :], label

good_tiles = [(im, 1) for im in load_ims('{}/good/'.format(data_path))]
bad_tiles = [(im, 0) for im in load_ims('{}/bad/'.format(data_path), limit=len(good_tiles))]
random.shuffle(good_tiles)
random.shuffle(bad_tiles)
val_tiles = good_tiles[0:512] + bad_tiles[0:512]
train_tiles = good_tiles[512:] + bad_tiles[512:]

print 'using {} train tiles, {} val tiles, {} good tiles, {} bad tiles'.format(len(train_tiles), len(val_tiles), len(good_tiles), len(bad_tiles))

latest_path = model_path + '/model_latest/model'
best_path = model_path + '/model_best/model'

m = model.Model()
session = tf.Session()
session.run(m.init_op)

best_accuracy = None

def epoch_to_learning_rate(epoch):
	if epoch < 100:
		return 1e-4
	elif epoch < 200:
		return 1e-5
	elif epoch < 300:
		return 1e-6
	else:
		return 1e-7

for epoch in xrange(400):
	start_time = time.time()
	train_losses = []
	for _ in xrange(8192):
		tiles = random.sample(train_tiles, model.BATCH_SIZE)
		examples = [extract_example(tile) for tile in tiles]
		_, loss = session.run([m.optimizer, m.loss], feed_dict={
			m.is_training: True,
			m.inputs: [example[0].astype('float32') / 255 for example in examples],
			m.targets: [[example[1]] for example in examples],
			m.learning_rate: epoch_to_learning_rate(epoch),
		})
		train_losses.append(loss)
	train_loss = numpy.mean(train_losses)
	train_time = time.time()

	val_losses = []
	val_accuracies = []
	for i in xrange(0, len(val_tiles), model.BATCH_SIZE):
		examples = [extract_example(tile) for tile in val_tiles[i:i+model.BATCH_SIZE]]
		loss, outputs = session.run([m.loss, m.outputs], feed_dict={
			m.is_training: False,
			m.inputs: [example[0].astype('float32') / 255 for example in examples],
			m.targets: [[example[1]] for example in examples],
		})
		val_losses.append(loss)
		for j in xrange(len(examples)):
			output = outputs[j] > 0.5
			gt = examples[j][1] > 0.5
			if output == gt:
				val_accuracies.append(1.0)
			else:
				val_accuracies.append(0.0)

	val_loss = numpy.mean(val_losses)
	val_accuracy = numpy.mean(val_accuracies)
	val_time = time.time()

	print 'iteration {}: train_time={}, val_time={}, train_loss={}, val_loss={}, val_acc={}/{}'.format(epoch, int(train_time - start_time), int(val_time - train_time), train_loss, val_loss, val_accuracy, best_accuracy)

	m.saver.save(session, latest_path)
	if best_accuracy is None or val_accuracy > best_accuracy:
		best_accuracy = val_accuracy
		m.saver.save(session, best_path)
