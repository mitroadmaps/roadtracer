import sys
sys.path.append('../lib')

from roadcnn import dataset
import model

import numpy
from PIL import Image
import random
import scipy.ndimage
import subprocess
import tensorflow as tf
import time

sat_path = sys.argv[1]
model_path = sys.argv[2]

best_path = model_path + '/model_best/model'
m = model.Model(big=True)
session = tf.Session()
m.saver.restore(session, best_path)

def apply_model(sat):
	output = numpy.zeros((sat.shape[0], sat.shape[1]), dtype='uint8')
	for x in range(0, sat.shape[0] - 2048, 512) + [sat.shape[0] - 2048]:
		for y in range(0, sat.shape[1] - 2048, 512) + [sat.shape[1] - 2048]:
			conv_input = sat[x:x+2048, y:y+2048, :].astype('float32') / 255.0
			conv_output = session.run(m.outputs, feed_dict={
				m.is_training: False,
				m.inputs: [conv_input],
			})[0, :, :]
			startx = (2048 - 512) / 2
			endx = (2048 + 512) / 2
			starty = (2048 - 512) / 2
			endy = (2048 + 512) / 2
			if x == 0:
				startx = 0
			elif x >= sat.shape[0] - 2048 - 512 - 1:
				endx = 2048
			if y == 0:
				starty = 0
			elif y >= sat.shape[1] - 2048 - 512 - 1:
				endy = 2048
			output[x+startx:x+endx, y+starty:y+endy] = conv_output[startx:endx, starty:endy] * 255.0
	return output

def get_test_tiles():
	regions = ['boston', 'new york', 'chicago', 'la', 'toronto', 'denver', 'kansas city', 'san diego', 'pittsburgh', 'montreal', 'vancouver', 'tokyo', 'saltlakecity', 'paris', 'amsterdam']
	tiles = {}
	for region in regions:
		if region == 'chicago':
			s = (-1, -2)
		elif region == 'boston':
			s = (1, -1)
		else:
			s = (-1, -1)
		im = numpy.zeros((8192, 8192, 3), dtype='uint8')
		for x in xrange(2):
			for y in xrange(2):
				fname = '{}/{}_{}_{}_sat.png'.format(sat_path, region, s[0] + x, s[1] + y)
				sat = scipy.ndimage.imread(fname)[:, :, 0:3]
				im[y*4096:y*4096+4096, x*4096:x*4096+4096, :] = sat
		tiles[region] = im
	return tiles

tiles = get_test_tiles()
for region, sat in tiles.items():
	output = apply_model(sat)
	Image.fromarray(output).save('outputs/{}.png'.format(region))
