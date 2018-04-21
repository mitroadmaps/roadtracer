from discoverlib import geom
from discoverlib import graph
from discoverlib import rdp

import numpy
import random

MIN_GRAPH_DISTANCE = 166
MAX_STRAIGHT_DISTANCE = 83
RDP_EPSILON = 2

def get_connections(g, im, limit=None):
	edge_im = -numpy.ones(im.shape, dtype='int32')
	for edge in g.edges:
		for p in geom.draw_line(edge.src.point, edge.dst.point, geom.Point(edge_im.shape[0], edge_im.shape[1])):
			edge_im[p.x, p.y] = edge.id

	road_segments, _ = graph.get_graph_road_segments(g)
	random.shuffle(road_segments)
	best_rs = None
	seen_vertices = set()
	proposed_connections = []
	for rs in road_segments:
		for vertex, opp in [(rs.src(), rs.point_at_factor(10)), (rs.dst(), rs.point_at_factor(rs.length() - 10))]:
			if len(vertex.out_edges) >= 2 or vertex in seen_vertices:
				continue
			seen_vertices.add(vertex)

			vertex_distances = get_vertex_distances(vertex, MIN_GRAPH_DISTANCE)
			edge, path = get_shortest_path(im, vertex.point, opp, edge_im, g, vertex_distances)
			if edge is not None:
				proposed_connections.append({
					'src': vertex.id,
					'edge': edge.id,
					'pos': edge.closest_pos(path[-1]).distance,
					'path': rdp.rdp([(p.x, p.y) for p in path], RDP_EPSILON),
				})
		if limit is not None and len(proposed_connections) >= limit:
			break

	return proposed_connections

def insert_connections(g, connections):
	split_edges = {} # map from edge to (split pos, new edge before pos, new edge after pos)
	for idx, connection in enumerate(connections):
		# figure out which current edge the connection intersects
		edge = g.edges[connection['edge']]
		path = [geom.Point(p[0], p[1]) for p in connection['path']]
		intersection_point = path[-1]
		while edge in split_edges:
			our_pos = edge.closest_pos(intersection_point).distance
			if our_pos < split_edges[edge]['pos']:
				edge = split_edges[edge]['before']
			else:
				edge = split_edges[edge]['after']

		# add path vertices
		prev_vertex = g.vertices[connection['src']]
		for point in path[1:]:
			vertex = g.add_vertex(point)
			edge1, edge2 = g.add_bidirectional_edge(prev_vertex, vertex)
			edge1.phantom = True
			edge1.connection_idx = idx
			edge2.phantom = True
			edge2.connection_idx = idx
			prev_vertex = vertex

		# split the edge
		new_vertex = prev_vertex
		for edge in [edge, edge.get_opposite_edge()]:
			split_pos = edge.closest_pos(intersection_point).distance
			split_edges[edge] = {
				'pos': split_pos,
				'before': g.add_edge(edge.src, new_vertex),
				'after': g.add_edge(new_vertex, edge.dst),
			}

	# remove extraneous edges
	filter_edges = set([edge for edge in split_edges.keys()])
	g = g.filter_edges(filter_edges)
	return g

def get_vertex_distances(src, max_distance):
	vertex_distances = {}

	seen_vertices = set()
	distances = {}
	distances[src] = 0
	while len(distances) > 0:
		closest_vertex = None
		closest_distance = None
		for vertex, distance in distances.items():
			if closest_vertex is None or distance < closest_distance:
				closest_vertex = vertex
				closest_distance = distance

		del distances[closest_vertex]
		vertex_distances[closest_vertex] = closest_distance
		seen_vertices.add(closest_vertex)
		if closest_distance > max_distance:
			break

		for edge in closest_vertex.out_edges:
			vertex = edge.dst
			if hasattr(edge, 'cost'):
				distance = closest_distance + edge.cost
			else:
				distance = closest_distance + edge.segment().length()
			if vertex not in seen_vertices and (vertex not in distances or distance < distances[vertex]):
				distances[vertex] = distance

	return vertex_distances

def get_shortest_path(im, src, opp, edge_im, g, vertex_distances):
	r = src.bounds().add_tol(MAX_STRAIGHT_DISTANCE)
	r = geom.Rectangle(geom.Point(0, 0), geom.Point(im.shape[0], im.shape[1])).clip_rect(r)
	seen_points = set()
	distances = {}
	prev = {}
	dst_edge = None
	dst_point = None

	distances[src] = 0
	while len(distances) > 0:
		closest_point = None
		closest_distance = None
		for point, distance in distances.items():
			if closest_point is None or distance < closest_distance:
				closest_point = point
				closest_distance = distance

		del distances[closest_point]
		seen_points.add(closest_point)
		if edge_im[closest_point.x, closest_point.y] >= 0:
			edge = g.edges[edge_im[closest_point.x, closest_point.y]]
			src_distance = vertex_distances.get(edge.src, MIN_GRAPH_DISTANCE)
			dst_distance = vertex_distances.get(edge.dst, MIN_GRAPH_DISTANCE)
			if src_distance + closest_point.distance(edge.src.point) >= MIN_GRAPH_DISTANCE and dst_distance + closest_point.distance(edge.dst.point) >= MIN_GRAPH_DISTANCE:
				dst_edge = edge
				dst_point = closest_point
				break

		for offset in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
			adj_point = closest_point.add(geom.Point(offset[0], offset[1]))
			if r.contains(adj_point) and adj_point not in seen_points and src.distance(adj_point) < opp.distance(adj_point):
				distance = closest_distance + 1 + (1 - im[adj_point.x, adj_point.y])
				if adj_point not in distances or distance < distances[adj_point]:
					distances[adj_point] = distance
					prev[adj_point] = closest_point

	if dst_edge is None:
		return None, None

	path = []
	point = dst_point
	while point != src:
		path.append(point)
		point = prev[point]
	path.append(src)
	path.reverse()

	return dst_edge, path
