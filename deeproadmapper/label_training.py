import sys
sys.path.append('../lib')

from discoverlib import graph
from discoverlib import geom
import get_connections
import label_gt
import scipy.ndimage
from PIL import Image
import numpy
from multiprocessing import Pool
import subprocess
import os

sat_path = sys.argv[1]
graph_path = sys.argv[2]
out_im_path = sys.argv[3]
out_graph_path = sys.argv[4]
dst_path = sys.argv[5]

regions = [fname.split('.png')[0] for fname in os.listdir(out_im_path) if '.png' in fname]

def process(region):
	print region
	city = region.split('_')[0]
	offset_x, offset_y = int(region.split('_')[1]), int(region.split('_')[2])

	sat = scipy.ndimage.imread('{}/{}_sat.png'.format(sat_path, region))
	sat = sat.swapaxes(0, 1)
	im = scipy.ndimage.imread('{}/{}.png'.format(out_im_path, region)).astype('float32') / 255.0
	im = im.swapaxes(0, 1)
	g = graph.read_graph('{}/{}.graph'.format(out_graph_path, region))
	g_idx = g.edgeIndex()

	gt = graph.read_graph('{}/{}.graph'.format(graph_path, city))
	offset = geom.Point(offset_x * 4096, offset_y * 4096)
	for vertex in gt.vertices:
		vertex.point = vertex.point.sub(offset)

	gt_idx = gt.edgeIndex()

	connections = get_connections.get_connections(g, im, limit=512)
	good, bad = label_gt.label_connections(gt, g, connections)
	for i, connection in enumerate(good):
		label_gt.write_connection(sat, im, g_idx, connection, '{}/good/{}_{}'.format(dst_path, region, i))
	for i, connection in enumerate(bad):
		label_gt.write_connection(sat, im, g_idx, connection, '{}/bad/{}_{}'.format(dst_path, region, i))
p = Pool()
p.map(process, regions)
p.close()
