import sys
sys.path.append('../lib')

from discoverlib import graph, geom

BRANCH_THRESHOLD = 15
LOOP_THRESHOLD = 50

in_fname = sys.argv[1]
out_fname = sys.argv[2]

g = graph.read_graph(in_fname)

bad_edges = set()
merge_vertices = {}
merge_groups = []

road_segments, _ = graph.get_graph_road_segments(g)
edge_index = g.edgeIndex()

# prune short branches
for rs in road_segments:
	if (len(rs.dst().out_edges) < 2 or len(rs.src().out_edges) < 2) and rs.length() < BRANCH_THRESHOLD:
		for edge in rs.edges:
			bad_edges.add(edge)

class Group(object):
	def __init__(self):
		self.l = []

	def add(self, x):
		if x not in self.l:
			self.l.append(x)

	def update(self, other):
		for x in other.l:
			self.add(x)

	def head(self):
		return self.l[0]

	def __iter__(self):
		return iter(self.l)

	def __len__(self):
		return len(self.l)

# merge short loops
for rs in road_segments:
	if rs.length() < LOOP_THRESHOLD:
		if rs.src() in merge_vertices and rs.dst() in merge_vertices:
			group = merge_vertices[rs.src()]
			dst_group = merge_vertices[rs.dst()]
			if group != dst_group:
				group.update(dst_group)
				for vertex in dst_group:
					merge_vertices[vertex] = group
		elif rs.src() in merge_vertices:
			group = merge_vertices[rs.src()]
			group.add(rs.dst())
			merge_vertices[rs.dst()] = group
		elif rs.dst() in merge_vertices:
			group = merge_vertices[rs.dst()]
			group.add(rs.src())
			merge_vertices[rs.src()] = group
		else:
			group = Group()
			group.add(rs.src())
			group.add(rs.dst())
			merge_vertices[rs.src()] = group
			merge_vertices[rs.dst()] = group
			merge_groups.append(group)
		for edge in rs.edges:
			merge_vertices[edge.src] = group
			merge_vertices[edge.dst] = group
			group.add(edge.src)
			group.add(edge.dst)

def get_avg(group):
	point_sum = geom.Point(0, 0)
	for vertex in group:
		point_sum = point_sum.add(vertex.point)
	return point_sum.scale(1.0 / len(group))

ng = graph.Graph()
vertex_map = {}

def get_vertex(vertex):
	if vertex in merge_vertices:
		group = merge_vertices[vertex]
		group_head = group.head()
		if group_head not in vertex_map:
			vertex_map[group_head] = ng.add_vertex(get_avg(group))
		return vertex_map[group_head]
	else:
		if vertex not in vertex_map:
			vertex_map[vertex] = ng.add_vertex(vertex.point)
		return vertex_map[vertex]

for edge in g.edges:
	if edge in bad_edges:
		continue
	src = get_vertex(edge.src)
	dst = get_vertex(edge.dst)
	if src == dst:
		continue
	ng.add_edge(src, dst)

ng.save(out_fname)
