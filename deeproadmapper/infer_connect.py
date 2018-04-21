import sys
sys.path.append('../lib')

from discoverlib import geom, graph
import label_gt
import get_connections
import model_connect as model

import json
import numpy
import os
from PIL import Image
import random
import scipy.ndimage
import subprocess
import tensorflow as tf
import time

model_path = sys.argv[1]
sat_path = sys.argv[2]
outim_path = sys.argv[3]
outgraph_path = sys.argv[4]
dst_path = sys.argv[5]

sat = scipy.ndimage.imread(sat_path)
sat = sat.swapaxes(0, 1)
outim = scipy.ndimage.imread(outim_path).astype('float32') / 255.0
outim = outim.swapaxes(0, 1)
g = graph.read_graph(outgraph_path)
g_idx = g.edgeIndex()

print 'loading model'
best_path = model_path + '/model_best/model'
m = model.Model()
session = tf.Session()
m.saver.restore(session, best_path)

print 'getting connections'
connections = get_connections.get_connections(g, outim)

print 'selecting connections'
selected_connections = []
for idx, connection in enumerate(connections):
	if idx % 128 == 0:
		print '... {}/{}'.format(idx, len(connections))

	im = label_gt.prepare_connection(sat, outim, g_idx, connection, size=256)
	output = session.run(m.outputs, feed_dict={
		m.is_training: False,
		m.inputs: [im.astype('float32') / 255],
	})[0]
	if output > 0.5:
		selected_connections.append(connection)

print 'selected {} of {} connections'.format(len(selected_connections), len(connections))
g = get_connections.insert_connections(g, selected_connections)
g.save(dst_path)
