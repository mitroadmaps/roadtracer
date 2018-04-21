import sys
sys.path.append('../lib')

from discoverlib import geom, graph
import os

region = sys.argv[1]
in_fname = sys.argv[2]
out_fname = sys.argv[3]

if region == 'boston':
	offset = geom.Point(4096, -4096)
elif region == 'chicago':
	offset = geom.Point(-4096, -8192)
else:
	offset = geom.Point(-4096, -4096)
g = graph.read_graph(in_fname)
for vertex in g.vertices:
	vertex.point = vertex.point.add(offset)
g.save(out_fname)
