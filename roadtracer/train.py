import sys
sys.path.append('../lib')

from discoverlib import geom, graph
import model
import model_utils
import tileloader
import infer

from collections import deque
import numpy
import math
import os
import os.path
from PIL import Image
import random
import scipy.ndimage
import tensorflow as tf
import time

import argparse
parser = argparse.ArgumentParser(description='Train a RoadTracer model.')
parser.add_argument('modelpath', help='path to save model')
parser.add_argument('--t', help='tiles/imagery path')
parser.add_argument('--g', help='graph path')
parser.add_argument('--j', help='path to directory containing pytiles.json/starting_locations.json')
args = parser.parse_args()

if args.t: tileloader.tile_dir = args.t
if args.g: tileloader.graph_dir = args.g
if args.j:
	tileloader.pytiles_path = os.path.join(args.j, 'pytiles.json')
	tileloader.startlocs_path = os.path.join(args.j, 'starting_locations.json')

MAX_PATH_LENGTH = 8192
SEGMENT_LENGTH = 20
PARALLEL_TILES = 256
SAVE_ITERATIONS = 1024
PATHS_PER_TILE_AXIS = 2
TILE_MODE = 'sat'
SAVE_EXAMPLES = True
DETECT_MODE = 'normal'
MODEL_BASE = args.modelpath
WINDOW_SIZE = 256
FOLLOW_TARGETS = False
THRESHOLD = 0.4
SINGLE_ANGLE_TARGET = False

PARALLEL_PATHS = PARALLEL_TILES * PATHS_PER_TILE_AXIS * PATHS_PER_TILE_AXIS

def epoch_to_learning_rate(epoch):
	if epoch < 100:
		return 1e-5
	elif epoch < 200:
		return 1e-6
	elif epoch < 300:
		return 1e-7
	else:
		return 1e-8

tiles = tileloader.Tiles(PATHS_PER_TILE_AXIS, SEGMENT_LENGTH, PARALLEL_TILES, TILE_MODE)
tiles.prepare_training()
test_tile_data = tiles.get_test_tile_data()

# initialize model and session
print 'initializing model'
m = model.Model(tiles.num_input_channels())
session = tf.Session()
model_path = MODEL_BASE + '/model_latest/model'
best_path = MODEL_BASE + '/model_best/model'
if os.path.isfile(model_path + '.meta'):
	print '... loading existing model'
	m.saver.restore(session, model_path)
else:
	print '... initializing a new model'
	session.run(m.init_op)

# initialize subtiles
subtiles = []
for tile in tiles.train_tiles:
	big_rect = geom.Rectangle(
		tile.scale(4096),
		tile.add(geom.Point(1, 1)).scale(4096)
	)
	for offset in [geom.Point(0, 0), geom.Point(0, 2048), geom.Point(2048, 0), geom.Point(2048, 2048)]:
		start = big_rect.start.add(offset)
		search_rect = geom.Rectangle(start, start.add(geom.Point(2048, 2048)))
		search_rect = search_rect.add_tol(-WINDOW_SIZE/2)

		starting_locations = tiles.all_starting_locations['{}_{}_{}'.format(tile.region, tile.x, tile.y)]
		starting_locations = [loc for loc in starting_locations if search_rect.add_tol(-WINDOW_SIZE/2).contains(loc[0]['point'])]
		if len(starting_locations) < 5:
			continue

		subtiles.append({
			'region': tile.region,
			'rect': big_rect,
			'search_rect': search_rect,
			'cache': tiles.cache,
			'starting_locations': starting_locations,
			'gc': tiles.gcs[tile.region],
			'edge_counts': {},
		})

print 'extracted {} subtiles from {} tiles (missing {})'.format(len(subtiles), len(tiles.train_tiles), 4*len(tiles.train_tiles) - len(subtiles))

# initialize paths, one per subtile
print 'loading initial paths'
paths = []
for i, subtile in enumerate(subtiles):
	start_loc = random.choice(subtile['starting_locations'])
	paths.append(model_utils.Path(subtile['gc'], subtile, start_loc=start_loc))
num_sets = (len(paths) + PARALLEL_PATHS - 1) / PARALLEL_PATHS

best_accuracy = None
angle_losses = []
detect_losses = []
action_losses = []
losses = []

def vector_to_action(angle_outputs, stop_outputs):
	x = numpy.zeros((64,), dtype='float32')
	if stop_outputs[0] > THRESHOLD:
		x[numpy.argmax(angle_outputs)] = 1
	return x

def action_to_vector(v):
	angle_outputs = numpy.zeros((64,), dtype='float32')
	action_outputs = numpy.zeros((2,), dtype='float32')
	count = 0
	for i in xrange(len(v)):
		if v[i] > 0.9:
			count += 1
	if count == 0:
		action_outputs[1] = 1
	else:
		action_outputs[0] = 1
		if SINGLE_ANGLE_TARGET:
			for i in xrange(len(v)):
				if v[i] > 0.9:
					angle_outputs[i] = 1.0 / count
		else:
			angle_outputs[:] = v[:]
	return angle_outputs, action_outputs

outer_it = 0

for outer_it in xrange(outer_it+1, 400):
	set_id = outer_it % num_sets
	start_idx = set_id * PARALLEL_PATHS
	end_idx = min(start_idx + PARALLEL_PATHS, len(paths))
	if end_idx - start_idx < PARALLEL_PATHS / 2:
		raise Exception('last set has only {} paths, but PARALLEL_PATHS={}'.format(end_idx - start_idx, PARALLEL_PATHS))

	times = {
		'prepare': 0,
		'train': 0,
		'save': 0,
		'extend': 0,
		'train_total': 0,
		'test_total': 0,
	}
	start_time = time.time()

	for path_it in xrange(2048):
		stage_time = time.time()
		if path_it % SAVE_ITERATIONS == 0 and False:
			print 'begin step {}, {} ({}...{}/{})'.format(outer_it, path_it, start_idx, end_idx, len(paths))
		path_indices = random.sample(range(start_idx, end_idx), model.BATCH_SIZE)

		# prepare path inputs and target angles
		batch_extension_vertices = []
		batch_inputs = []
		batch_detect_targets = []
		batch_angle_targets = numpy.zeros((model.BATCH_SIZE, 64), 'float32')
		batch_action_targets = numpy.zeros((model.BATCH_SIZE, 2), 'float32')
		for i in xrange(len(path_indices)):
			path_idx = path_indices[i]

			extension_vertex = paths[path_idx].pop()
			if extension_vertex is None or len(paths[path_idx].graph.vertices) >= MAX_PATH_LENGTH:
				start_loc = random.choice(subtiles[path_idx]['starting_locations'])
				paths[path_idx] = model_utils.Path(subtiles[path_idx]['gc'], subtiles[path_idx], start_loc=start_loc)
				extension_vertex = paths[path_idx].pop()

			path_input, path_detect_target = model_utils.make_path_input(paths[path_idx], extension_vertex, SEGMENT_LENGTH, detect_mode=DETECT_MODE, window_size=WINDOW_SIZE)
			batch_extension_vertices.append(extension_vertex)
			batch_inputs.append(path_input)
			batch_detect_targets.append(path_detect_target)

			targets = model_utils.compute_targets_by_best(paths[path_idx], extension_vertex, SEGMENT_LENGTH)
			angle_targets, action_targets = action_to_vector(targets)
			batch_angle_targets[i, :] = angle_targets
			batch_action_targets[i, :] = action_targets

		times['prepare'] += time.time() - stage_time
		stage_time = time.time()

		# train model
		feed_dict = {
			m.is_training: True,
			m.inputs: batch_inputs,
			m.angle_targets: batch_angle_targets,
			m.action_targets: batch_action_targets,
			m.detect_targets: batch_detect_targets,
			m.learning_rate: epoch_to_learning_rate(outer_it),
		}
		batch_angle_outputs, batch_action_outputs, batch_detect_outputs, angle_loss, detect_loss, action_loss, loss, _ = session.run([m.angle_outputs, m.action_outputs, m.detect_outputs, m.angle_loss, m.detect_loss, m.action_loss, m.loss, m.optimizer], feed_dict=feed_dict)

		angle_losses.append(angle_loss)
		detect_losses.append(detect_loss)
		action_losses.append(action_loss)
		losses.append(loss)

		times['train'] += time.time() - stage_time
		stage_time = time.time()

		if SAVE_EXAMPLES and start_idx in path_indices:
			x = path_indices.index(start_idx)
			fname = '/home/ubuntu/data/{}_{}_{}_'.format(path_indices[x], outer_it, path_it)
			model_utils.make_path_input(paths[path_indices[x]], batch_extension_vertices[x], SEGMENT_LENGTH, fname=fname, angle_targets=batch_angle_targets[x, :], angle_outputs=batch_angle_outputs[x, :], detect_output=batch_detect_outputs[x, :, :, 0], detect_mode=DETECT_MODE, window_size=WINDOW_SIZE)

			with open(fname + 'meta.txt', 'w') as f:
				f.write('action={}, angle_bucket={}\n\nactions: {}\nangles: \n'.format(
					numpy.argmax(batch_action_outputs[x, :]),
					numpy.argmax(batch_angle_outputs[x, :]),
					batch_action_outputs[x, :],
					batch_angle_outputs[x, :]
				))

		times['save'] += time.time() - stage_time
		stage_time = time.time()

		# extend paths based on angle outputs
		for i in xrange(len(path_indices)):
			path_idx = path_indices[i]
			if FOLLOW_TARGETS == True:
				x = vector_to_action(batch_angle_targets[i, :], batch_action_targets[i, :])
			elif FOLLOW_TARGETS == 'partial':
				# (a) always use stop_targets instead of stop_outputs
				# (b) if we are far away from graph, use angle_targets, otherwise use angle_outputs
				extension_vertex = batch_extension_vertices[i]
				if extension_vertex.edge_pos is None or extension_vertex.edge_pos.point().distance(extension_vertex.point) > SEGMENT_LENGTH * 2:
					x = vector_to_action(batch_angle_targets[i, :], batch_action_targets[i, :])
				else:
					x = vector_to_action(batch_angle_outputs[i, :], batch_action_targets[i, :])
			elif FOLLOW_TARGETS == 'npartial':
				# always move if gt says to move
				if batch_action_outputs[i, 0] > THRESHOLD:
					x = vector_to_action(batch_angle_outputs[i, :], batch_action_outputs[i, :])
				else:
					x = vector_to_action(batch_angle_outputs[i, :], batch_action_targets[i, :])
			elif FOLLOW_TARGETS == False:
				x = vector_to_action(batch_angle_outputs[i, :], batch_action_outputs[i, :])
			else:
				raise Exception('invalid FOLLOW_TARGETS setting {}'.format(FOLLOW_TARGETS))

			nvertex = len(paths[path_idx].graph.vertices)
			paths[path_idx].push(batch_extension_vertices[i], x, SEGMENT_LENGTH)
			if len(paths[path_idx].graph.vertices) > nvertex:
				pos1 = paths[path_idx].graph.vertices[-1].edge_pos
				pos2 = paths[path_idx].graph.vertices[-2].edge_pos
				if pos1 is not None and pos2 is not None and pos1.edge != pos2.edge:
					subtiles[path_idx]['edge_counts'][pos1.edge.id] = subtiles[path_idx]['edge_counts'].get(pos1.edge.id, 0) + 1

		times['extend'] += time.time() - stage_time
		stage_time = time.time()

		if path_it % SAVE_ITERATIONS == 0:
			print 'step {},{} train: angle_loss={}, detect_loss={}, action_loss={}, loss={}'.format(outer_it, path_it, numpy.mean(angle_losses), numpy.mean(detect_losses), numpy.mean(action_losses), numpy.mean(losses))
			del angle_losses[:]
			del detect_losses[:]
			del action_losses[:]
			del losses[:]
			m.saver.save(session, model_path)

		times['save'] += time.time() - stage_time
		stage_time = time.time()

	times['train_total'] += time.time() - start_time
	start_time = time.time()

	# run test
	if test_tile_data is not None:
		test_paths = []
		if not isinstance(test_tile_data, list):
			test_tile_data = [test_tile_data]
		for t in test_tile_data:
			test_paths.append(model_utils.Path(t['gc'], t, start_loc=t['starting_locations'][1]))
		angle_loss, detect_loss, action_loss, loss, path_length, accuracy = infer.eval(test_paths, m, session, max_path_length=2048, segment_length=SEGMENT_LENGTH, follow_targets=True, max_batch_size=model.BATCH_SIZE, window_size=WINDOW_SIZE, verbose=False)
		print '*** TEST ***: angle_loss={}, detect_loss={}, action_loss={}, loss={}, len={}, accuracy={}/{}'.format(angle_loss, detect_loss, action_loss, loss, path_length, accuracy, best_accuracy)
		if best_accuracy is None or accuracy > best_accuracy:
			best_accuracy = accuracy
			m.saver.save(session, best_path)

	times['test_total'] += time.time() - start_time
	print times
