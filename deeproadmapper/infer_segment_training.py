import sys
sys.path.append('../lib')

from roadcnn import dataset
import model_segment as model

import numpy
from PIL import Image
import random
import scipy.ndimage
import subprocess
import tensorflow as tf
import time

sat_path = sys.argv[1]
osm_path = sys.argv[2]
model_path = sys.argv[3]
out_path = sys.argv[4]

print 'loading train tiles'
train_tiles = dataset.load_tiles(sat_path, osm_path, 'train')
best_path = model_path + '/model_best/model'
m = model.Model(mode=0, big=True)
session = tf.Session()
m.saver.restore(session, best_path)

def apply2(tiles, out_dir='out'):
	for i, tile in enumerate(tiles):
		print '{}/{}'.format(i, len(tiles))
		region, sat, _ = tile
		output = session.run(m.outputs, feed_dict={
			m.is_training: False,
			m.inputs: [sat.astype('float32') / 255.0],
		})[0, :, :]
		Image.fromarray((output * 255).astype('uint8')).save('{}/{}.png'.format(out_dir, region))

apply2(train_tiles, out_dir=out_path)
