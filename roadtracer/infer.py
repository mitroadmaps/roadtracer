import sys
sys.path.append('../lib')

from discoverlib import geom, graph
import model
import model_utils
import tileloader

import numpy
import math
import os.path
from PIL import Image
import random
import scipy.ndimage
import tensorflow as tf
import time

MAX_PATH_LENGTH = 500000
SEGMENT_LENGTH = 20
PATHS_PER_TILE_AXIS = 1
TILE_MODE = 'sat'
EXISTING_GRAPH_FNAME = None
DETECT_MODE = 'normal'
THRESHOLD_BRANCH = 0.4
THRESHOLD_FOLLOW = 0.4
WINDOW_SIZE = 256
SAVE_EXAMPLES = False
FOLLOW_TARGETS = False

REGION = 'chicago'
TILE_SIZE = 4096
TILE_START = geom.Point(-1, -2).scale(TILE_SIZE)
TILE_END = TILE_START.add(geom.Point(2, 2).scale(TILE_SIZE))

USE_TL_LOCATIONS = True
MANUAL_RELATIVE = geom.Point(-1, -2).scale(TILE_SIZE)
MANUAL_POINT1 = geom.Point(2560, 522)
MANUAL_POINT2 = geom.Point(2592, 588)

def vector_to_action(angle_outputs, stop_outputs, threshold):
	x = numpy.zeros((64,), dtype='float32')
	if stop_outputs[0] > threshold:
		x[numpy.argmax(angle_outputs)] = stop_outputs[0]
	return x

def action_to_vector(v):
	angle_outputs = numpy.zeros((64,), dtype='float32')
	stop_outputs = numpy.zeros((2,), dtype='float32')
	count = 0
	for i in xrange(len(v)):
		if v[i] > 0.9:
			count += 1
	if count == 0:
		stop_outputs[1] = 1
	else:
		stop_outputs[0] = 1
		for i in xrange(len(v)):
			if v[i] > 0.9:
				angle_outputs[i] = 1.0 / count
	return angle_outputs, stop_outputs

def fix_outputs(batch_angle_outputs, batch_stop_outputs):
	if batch_angle_outputs.shape[1] == 64:
		return batch_angle_outputs, batch_stop_outputs
	elif batch_angle_outputs.shape[1] == 65:
		fixed_stop_outputs = numpy.zeros((batch_angle_outputs.shape[0], 2), dtype='float32')
		for i in xrange(batch_angle_outputs.shape[0]):
			if numpy.argmax(batch_angle_outputs[i, :]) == 64:
				fixed_stop_outputs[i, 1] = 1
			else:
				fixed_stop_outputs[i, 0] = 1
		return batch_angle_outputs[:, 0:64], fixed_stop_outputs
	else:
		raise Exception("bad angle_outputs length={}".format(len(angle_outputs)))

def score_accuracy(stop_targets, angle_targets, stop_outputs, angle_outputs, threshold, action_only=False):
	target_action = stop_targets[0] > threshold
	output_action = stop_outputs[0] > threshold
	if target_action != output_action:
		accuracy = 0.0
	elif not target_action:
		accuracy = 1.0
	elif action_only:
		accuracy =  1.0
	else:
		target_angle = numpy.argmax(angle_targets)
		output_angle = numpy.argmax(angle_outputs)
		angle_distance = abs(target_angle - output_angle)
		if angle_distance > 32:
			angle_distance = 64 - angle_distance
		if angle_distance > 16:
			accuracy = 0.0
		else:
			accuracy = 1.0 - float(angle_distance) / 16
	return accuracy

def eval(paths, m, session, max_path_length=MAX_PATH_LENGTH, segment_length=SEGMENT_LENGTH, save=False, follow_targets=False, compute_targets=True, max_batch_size=model.BATCH_SIZE, window_size=WINDOW_SIZE, verbose=True, threshold_override=False):
	angle_losses = []
	detect_losses = []
	stop_losses = []
	losses = []
	accuracies = []
	path_lengths = {path_idx: 0 for path_idx in xrange(len(paths))}

	last_time = None
	big_time = None

	for len_it in xrange(99999999):
		if len_it % 500 == 0 and verbose:
			print 'it {}'.format(len_it)
			big_time = time.time()
		path_indices = []
		extension_vertices = []
		for path_idx in xrange(len(paths)):
			if path_lengths[path_idx] >= max_path_length:
				continue
			extension_vertex = paths[path_idx].pop()
			if extension_vertex is None:
				continue
			path_indices.append(path_idx)
			path_lengths[path_idx] += 1
			extension_vertices.append(extension_vertex)

			if len(path_indices) >= max_batch_size:
				break

		if len(path_indices) == 0:
			break

		batch_inputs = []
		batch_detect_targets = []
		batch_angle_targets = numpy.zeros((len(path_indices), 64), 'float32')
		batch_stop_targets = numpy.zeros((len(path_indices), 2), 'float32')

		for i in xrange(len(path_indices)):
			path_idx = path_indices[i]

			path_input, path_detect_target = model_utils.make_path_input(paths[path_idx], extension_vertices[i], segment_length, window_size=window_size)
			batch_inputs.append(path_input)
			batch_detect_targets.append(path_detect_target)

			if compute_targets:
				targets = model_utils.compute_targets_by_best(paths[path_idx], extension_vertices[i], segment_length)
				angle_targets, stop_targets = action_to_vector(targets)
				batch_angle_targets[i, :] = angle_targets
				batch_stop_targets[i, :] = stop_targets

		feed_dict = {
			m.is_training: False,
			m.inputs: batch_inputs,
			m.angle_targets: batch_angle_targets,
			m.action_targets: batch_stop_targets,
			m.detect_targets: batch_detect_targets,
		}
		batch_angle_outputs, batch_stop_outputs, batch_detect_outputs, angle_loss, detect_loss, stop_loss, loss = session.run([m.angle_outputs, m.action_outputs, m.detect_outputs, m.angle_loss, m.detect_loss, m.action_loss, m.loss], feed_dict=feed_dict)
		angle_losses.append(angle_loss)
		detect_losses.append(detect_loss)
		stop_losses.append(stop_loss)
		losses.append(loss)
		batch_angle_outputs, batch_stop_outputs = fix_outputs(batch_angle_outputs, batch_stop_outputs)

		if save and len_it % 1 == 0:
			fname = '/home/ubuntu/data/{}_'.format(len_it)
			save_angle_targets = batch_angle_targets[0, :]
			if not compute_targets:
				save_angle_targets = None
			model_utils.make_path_input(paths[path_indices[0]], extension_vertices[0], segment_length, fname=fname, angle_targets=save_angle_targets, angle_outputs=batch_angle_outputs[0, :], detect_output=batch_detect_outputs[0, :, :, 0], window_size=window_size)

		for i in xrange(len(path_indices)):
			path_idx = path_indices[i]
			if len(extension_vertices[i].out_edges) >= 2:
				threshold = THRESHOLD_BRANCH
				mode = 'branch'
			else:
				threshold = THRESHOLD_FOLLOW
				mode = 'follow'
			if threshold_override:
				threshold = threshold_override

			if follow_targets == True:
				x = vector_to_action(batch_angle_targets[i, :], batch_stop_targets[i, :], threshold=threshold)
			elif follow_targets == 'partial':
				# (a) always use stop_targets instead of stop_outputs
				# (b) if we are far away from graph, use angle_targets, otherwise use angle_outputs
				extension_vertex = batch_extension_vertices[i]
				if extension_vertex.edge_pos is None or extension_vertex.edge_pos.point().distance(extension_vertex.point) > SEGMENT_LENGTH * 2:
					x = vector_to_action(batch_angle_targets[i, :], batch_stop_targets[i, :], threshold=threshold)
				else:
					x = vector_to_action(batch_angle_outputs[i, :], batch_stop_targets[i, :], threshold=threshold)
			elif follow_targets == 'npartial':
				# always move if gt says to move
				if batch_stop_outputs[i, 0] > threshold:
					x = vector_to_action(batch_angle_outputs[i, :], batch_stop_outputs[i, :], threshold=threshold)
				else:
					x = vector_to_action(batch_angle_outputs[i, :], batch_stop_targets[i, :], threshold=threshold)
			elif follow_targets == False:
				x = vector_to_action(batch_angle_outputs[i, :], batch_stop_outputs[i, :], threshold=threshold)
			else:
				raise Exception('invalid FOLLOW_TARGETS setting {}'.format(follow_targets))

			paths[path_idx].push(extension_vertices[i], x, segment_length, training=False, branch_threshold=0.01, follow_threshold=0.01)

			# score accuracy
			accuracy = score_accuracy(batch_stop_targets[i, :], batch_angle_targets[i, :], batch_stop_outputs[i, :], batch_angle_outputs[i, :], threshold)
			accuracies.append(accuracy)

	if save:
		paths[0].graph.save('out.graph')

	return numpy.mean(angle_losses), numpy.mean(detect_losses), numpy.mean(stop_losses), numpy.mean(losses), len_it, numpy.mean(accuracies)

def graph_filter(g, threshold=0.3, min_len=None):
	road_segments, _ = graph.get_graph_road_segments(g)
	bad_edges = set()
	for rs in road_segments:
		if min_len is not None and len(rs.edges) < min_len:
			bad_edges.update(rs.edges)
			continue
		probs = []
		if len(rs.edges) < 5 or True:
			for edge in rs.edges:
				if hasattr(edge, 'prob'):
					probs.append(edge.prob)
		else:
			for edge in rs.edges[2:-2]:
				if hasattr(edge, 'prob'):
					probs.append(edge.prob)
		if not probs:
			continue
		avg_prob = numpy.mean(probs)
		if avg_prob < threshold:
			bad_edges.update(rs.edges)
	print 'filtering {} edges'.format(len(bad_edges))
	ng = graph.Graph()
	vertex_map = {}
	for vertex in g.vertices:
		vertex_map[vertex] = ng.add_vertex(vertex.point)
	for edge in g.edges:
		if edge not in bad_edges:
			ng.add_edge(vertex_map[edge.src], vertex_map[edge.dst])
	return ng

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Run RoadTracer inference.')
	parser.add_argument('modelpath', help='trained model path')
	parser.add_argument('outname', help='output filename to save inferred road network graph')
	parser.add_argument('--s', help='stop threshold (default 0.4)', default=0.4)
	parser.add_argument('--r', help='region (default chicago)', default='chicago')
	parser.add_argument('--t', help='tiles/imagery path')
	parser.add_argument('--g', help='graph path')
	parser.add_argument('--j', help='path to directory containing pytiles.json/starting_locations.json')
	parser.add_argument('--e', help='existing graph to get starting locations from')
	parser.add_argument('--f', help='filter threshold to filter output edges (e.g. 0.75, default disabled)', default=0)
	args = parser.parse_args()
	model_path = args.modelpath
	output_fname = args.outname
	BRANCH_THRESHOLD = args.s
	FOLLOW_THRESHOLD = args.s
	REGION = args.r
	if REGION == 'boston':
		TILE_START = geom.Point(1, -1).scale(TILE_SIZE)
	elif REGION == 'chicago':
		TILE_START = geom.Point(-1, -2).scale(TILE_SIZE)
	else:
		TILE_START = geom.Point(-1, -1).scale(TILE_SIZE)
	TILE_END = TILE_START.add(geom.Point(2, 2).scale(TILE_SIZE))

	if args.t: tileloader.tile_dir = args.t
	if args.g: tileloader.graph_dir = args.g
	if args.j:
		tileloader.pytiles_path = os.path.join(args.j, 'pytiles.json')
		tileloader.startlocs_path = os.path.join(args.j, 'starting_locations.json')

	print 'reading tiles'
	#tileloader.use_vhr()
	tiles = tileloader.Tiles(PATHS_PER_TILE_AXIS, SEGMENT_LENGTH, 16, TILE_MODE)

	print 'initializing model'
	model.BATCH_SIZE = 1
	m = model.Model(tiles.num_input_channels())
	session = tf.Session()
	m.saver.restore(session, model_path)

	if EXISTING_GRAPH_FNAME is None:
		rect = geom.Rectangle(TILE_START, TILE_END)
		tile_data = tiles.get_tile_data(REGION, rect)

		if USE_TL_LOCATIONS:
			start_loc = random.choice(tile_data['starting_locations'])
		else:
			def match_point(p):
				best_pos = None
				best_distance = None
				for candidate in tile_data['gc'].edge_index.search(p.bounds().add_tol(32)):
					pos = candidate.closest_pos(p)
					distance = pos.point().distance(p)
					if best_pos is None or distance < best_distance:
						best_pos = pos
						best_distance = distance
				return best_pos
			pos1_point = MANUAL_POINT1.add(MANUAL_RELATIVE)
			pos1_pos = match_point(pos1_point)

			if MANUAL_POINT2:
				pos2_point = MANUAL_POINT2.add(MANUAL_RELATIVE)
				pos2_pos = match_point(pos2_point)
			else:
				next_positions = graph.follow_graph(pos1_pos, SEGMENT_LENGTH)
				pos2_point = next_positions[0].point()
				pos2_pos = next_positions[0]

			start_loc = [{
				'point': pos1_point,
				'edge_pos': pos1_pos,
			}, {
				'point': pos2_point,
				'edge_pos': pos2_pos,
			}]

		path = model_utils.Path(tile_data['gc'], tile_data, start_loc=start_loc)
	else:
		g = graph.read_graph(EXISTING_GRAPH_FNAME)
		r = g.bounds()
		tile_data = {
			'region': REGION,
			'rect': r.add_tol(WINDOW_SIZE/2),
			'search_rect': r,
			'cache': cache,
			'starting_locations': [],
		}
		path = model_utils.Path(None, tile_data, g=g)
		for vertex in g.vertices:
			path.prepend_search_vertex(vertex)

	compute_targets = SAVE_EXAMPLES or FOLLOW_TARGETS
	if args.e:
		ng = graph.read_graph(args.e)
		pg = graph.Graph()
		path = model_utils.Path(None, tile_data, g=pg)
		for edge in ng.edges:
			r = edge.segment().bounds().add_tol(100)
			nearby_edges = path.edge_rtree.intersection((r.start.x, r.start.y, r.end.x, r.end.y))
			if len(list(nearby_edges)) > 0:
				print 'skip {}'.format(edge.id)
				continue
			print 'process {}'.format(edge.id)
			v1 = pg.add_vertex(edge.src.point)
			v2 = pg.add_vertex(edge.dst.point)
			v1.edge_pos = None
			v2.edge_pos = None
			#path._add_bidirectional_edge(v1, v2)
			path.prepend_search_vertex(v1)
			path.prepend_search_vertex(v2)
			result = eval([path], m, session, save=False, compute_targets=compute_targets, follow_targets=FOLLOW_TARGETS)
	else:
		result = eval([path], m, session, save=SAVE_EXAMPLES, compute_targets=compute_targets, follow_targets=FOLLOW_TARGETS)
	print result
	if args.f > 0:
		path.graph = graph_filter(path.graph, threshold=0.75, min_len=8)
	path.graph.save(output_fname)
