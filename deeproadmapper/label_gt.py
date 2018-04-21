from discoverlib import geom
from discoverlib import graph
import get_connections

import numpy
from PIL import Image

PHANTOM_BASE = 8
PHANTOM_WEIGHT = 3
MAX_GT_DISTANCE = 20

def label_connections(gt_graph, inferred_graph, connections, threshold=25):
	g = inferred_graph.clone()
	get_connections.insert_connections(g, connections)
	edge_index = g.edgeIndex()
	gt_index = gt_graph.edgeIndex()
	for edge in g.edges:
		if hasattr(edge, 'phantom'):
			edge.cost = PHANTOM_BASE + PHANTOM_WEIGHT * edge.segment().length()

			# if it is far from ground truth edge, increase it's edge cost
			'''segment = edge.segment()
			midpoint = segment.start.add(segment.vector().scale(0.5))
			gt_distance = None
			for edge in gt_index.search(midpoint.bounds().add_tol(MAX_GT_DISTANCE)):
				d = edge.segment().distance(midpoint)
				if gt_distance is None or d < gt_distance:
					gt_distance = d
			if gt_distance is None or gt_distance > MAX_GT_DISTANCE:
				edge.cost *= 100'''

	# run dijkstra's on every ground truth road segment whose endpoints match
	# any connections that are traversed in such a search are marked "good"
	def match_point(point):
		best_vertex = None
		best_distance = None
		for edge in edge_index.search(point.bounds().add_tol(threshold)):
			if hasattr(edge, 'phantom'):
				continue
			for vertex in [edge.src, edge.dst]:
				d = point.distance(vertex.point)
				if d < threshold and (best_vertex is None or d < best_distance):
					best_vertex = vertex
					best_distance = d
		return best_vertex

	road_segments, _ = graph.get_graph_road_segments(gt_graph)
	good_connection_indices = set()
	for rs in road_segments:
		src = match_point(rs.src().point)
		dst = match_point(rs.dst().point)
		if src is None or dst is None:
			continue
		_, edge_path = graph.shortest_path(src, dst, max_distance=(PHANTOM_WEIGHT+1)*rs.length())
		if edge_path is None:
			continue

		for edge in edge_path:
			if hasattr(edge, 'connection_idx'):
				good_connection_indices.add(edge.connection_idx)

	good_connections = [connection for i, connection in enumerate(connections) if i in good_connection_indices]
	bad_connections = [connection for i, connection in enumerate(connections) if i not in good_connection_indices]
	return good_connections, bad_connections

def visualize_connection(sat, gt_idx, inferred_idx, connection, fname, good=True):
	path = [geom.Point(p[0], p[1]) for p in connection['path']]
	r = path[0].bounds()
	for p in path:
		r = r.extend(p)
	r = r.add_tol(128)
	r = geom.Rectangle(geom.Point(0, 0), geom.Point(sat.shape[0], sat.shape[1])).clip_rect(r)
	im = numpy.copy(sat[r.start.x:r.end.x, r.start.y:r.end.y])
	im_rect = geom.Rectangle(geom.Point(0, 0), geom.Point(im.shape[0], im.shape[1]))

	def color_point(p, color, tol=1):
		s = im_rect.clip(p.sub(geom.Point(tol, tol)))
		e = im_rect.clip(p.add(geom.Point(tol, tol)))
		im[s.x:e.x+1, s.y:e.y+1, :] = color

	# draw graph yellow
	for edge in inferred_idx.search(r.add_tol(32)):
		segment = edge.segment()
		start = segment.start.sub(r.start)
		end = segment.end.sub(r.start)
		for p in geom.draw_line(start, end, r.lengths()):
			color_point(p, [255, 255, 0])

	# draw connection red or green
	for i in xrange(len(path) - 1):
		start = path[i].sub(r.start)
		end = path[i + 1].sub(r.start)
		for p in geom.draw_line(start, end, r.lengths()):
			if good:
				color_point(p, [0, 255, 0])
			else:
				color_point(p, [255, 0, 0])

	# draw gt graph blue
	for edge in gt_idx.search(r.add_tol(32)):
		segment = edge.segment()
		start = segment.start.sub(r.start)
		end = segment.end.sub(r.start)
		for p in geom.draw_line(start, end, r.lengths()):
			color_point(p, [0, 0, 255], tol=0)

	Image.fromarray(im.swapaxes(0, 1)).save(fname)

def prepare_connection(sat, outim, inferred_idx, connection, size=320):
	path = [geom.Point(p[0], p[1]) for p in connection['path']]
	r = path[0].bounds()
	for p in path:
		r = r.extend(p)
	l = r.lengths()
	if l.x > 256 or l.y > 256:
		return
	s = geom.Point((size - l.x)/2, (size - l.y)/2)
	r = geom.Rectangle(r.start.sub(s), r.end.add(s))
	r = geom.Rectangle(geom.Point(0, 0), geom.Point(sat.shape[0], sat.shape[1])).clip_rect(r)
	l = r.lengths()
	im = numpy.zeros((size, size, 6), dtype='uint8')
	im[0:l.x, 0:l.y, 0:3] = sat[r.start.x:r.end.x, r.start.y:r.end.y, :]
	im[0:l.x, 0:l.y, 5] = outim[r.start.x:r.end.x, r.start.y:r.end.y]

	# draw graph
	for edge in inferred_idx.search(r.add_tol(32)):
		segment = edge.segment()
		start = segment.start.sub(r.start)
		end = segment.end.sub(r.start)
		for p in geom.draw_line(start, end, r.lengths()):
			im[p.x, p.y, 3] = 255

	# draw connection
	for i in xrange(len(path) - 1):
		start = path[i].sub(r.start)
		end = path[i + 1].sub(r.start)
		for p in geom.draw_line(start, end, r.lengths()):
			im[p.x, p.y, 4] = 255

	return im

def write_connection(sat, outim, inferred_idx, connection, fname):
	im = prepare_connection(sat, outim, inferred_idx, connection)
	Image.fromarray(im[:, :, 0:3].swapaxes(0, 1)).save(fname + '.sat.png')
	Image.fromarray(im[:, :, 3:6].swapaxes(0, 1)).save(fname + '.con.png')

def sample_points(path, sample_freq=5):
	samples = []
	remaining = 0
	for i in xrange(len(path) - 1):
		cur_point = path[i]
		next_point = path[i + 1]
		l = next_point.distance(cur_point)
		d = 0
		while l - d > remaining:
			d += remaining
			remaining = SAMPLE_FREQ
			samples.append(cur_point.add(next_point.sub(cur_point).scale(d / l)))
		remaining -= l - d
	return samples

def extract_features(im, connection):
	path = [geom.Point(p[0], p[1]) for p in connection['path']]
	samples = sample_points(path)
	softmax_scores = []
	distance_nonroad = []
	for point in samples:
		softmax_scores.append(im[point.x, point.y])
