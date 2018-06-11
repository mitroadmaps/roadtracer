from discoverlib import geom
from discoverlib import graph
import model_utils

import json
import numpy
import os
import random
import rtree
import scipy.ndimage
import time

tile_dir = '/data/imagery/'
graph_dir = '/data/graphs/'
pytiles_path = '/data/json/pytiles.json'
startlocs_path = '/data/json/starting_locations.json'
tile_size = 4096
window_size = 512
TRAINING_REGIONS = ['indianapolis', 'louisville', 'columbus', 'milwaukee', 'minneapolis', 'seattle', 'portland', 'sf', 'san antonio', 'vegas', 'phoenix', 'dallas', 'austin', 'san jose', 'houston', 'miami', 'tampa', 'orlando', 'atlanta', 'st louis', 'nashville', 'dc', 'baltimore', 'philadelphia', 'london']
REGIONS = TRAINING_REGIONS + ['chicago']

def load_tile(region, i, j, mode='all'):
	prefix = '{}/{}_{}_{}_'.format(tile_dir, region, i, j)
	sat_im = scipy.ndimage.imread(prefix + 'sat.png')
	if sat_im.shape == (tile_size, tile_size, 4):
		sat_im = sat_im[:, :, 0:3]
	return {
		'input': numpy.swapaxes(sat_im, 0, 1),
	}

def load_rect(region, rect, load_func=load_tile, mode='all'):
	# special case for fast load: rect is single tile
	if rect.start.x % tile_size == 0 and rect.start.y % tile_size == 0 and rect.end.x % tile_size == 0 and rect.end.y % tile_size == 0 and rect.end.x - rect.start.x == tile_size and rect.end.y - rect.start.y == tile_size:
		return load_func(region, rect.start.x / tile_size, rect.start.y / tile_size, mode=mode)

	tile_rect = geom.Rectangle(
		geom.Point(rect.start.x / tile_size, rect.start.y / tile_size),
		geom.Point((rect.end.x - 1) / tile_size + 1, (rect.end.y - 1) / tile_size + 1)
	)
	full_rect = geom.Rectangle(
		tile_rect.start.scale(tile_size),
		tile_rect.end.scale(tile_size)
	)
	full_ims = {}

	for i in xrange(tile_rect.start.x, tile_rect.end.x):
		for j in xrange(tile_rect.start.y, tile_rect.end.y):
			p = geom.Point(i - tile_rect.start.x, j - tile_rect.start.y).scale(tile_size)
			tile_ims = load_func(region, i, j, mode=mode)
			for k, im in tile_ims.iteritems():
				scale = tile_size / im.shape[0]
				if k not in full_ims:
					full_ims[k] = numpy.zeros((full_rect.lengths().x / scale, full_rect.lengths().y / scale, im.shape[2]), dtype='uint8')
				full_ims[k][p.x/scale:(p.x+tile_size)/scale, p.y/scale:(p.y+tile_size)/scale, :] = im

	crop_rect = geom.Rectangle(
		rect.start.sub(full_rect.start),
		rect.end.sub(full_rect.start)
	)
	for k in full_ims:
		scale = (full_rect.end.x - full_rect.start.x) / full_ims[k].shape[0]
		full_ims[k] = full_ims[k][crop_rect.start.x/scale:crop_rect.end.x/scale, crop_rect.start.y/scale:crop_rect.end.y/scale, :]
	return full_ims

class TileCache(object):
	def __init__(self, limit=128, mode='all'):
		self.limit = limit
		self.mode = mode
		self.cache = {}
		self.last_used = {}

	def reduce_to(self, limit):
		while len(self.cache) > limit:
			best_k = None
			best_used = None
			for k in self.cache:
				if best_k is None or self.last_used.get(k, 0) < best_used:
					best_k = k
					best_used = self.last_used.get(k, 0)
			del self.cache[best_k]

	def get(self, region, rect):
		k = '{}.{}.{}.{}.{}'.format(region, rect.start.x, rect.start.y, rect.end.x, rect.end.y)
		if k not in self.cache:
			self.reduce_to(self.limit - 1)
			self.cache[k] = load_rect(region, rect, mode=self.mode)
		self.last_used[k] = time.time()
		return self.cache[k]

	def get_window(self, region, big_rect, small_rect):
		big_dict = self.get(region, big_rect)
		small_dict = {}
		for k, v in big_dict.items():
			small_dict[k] = v[small_rect.start.x:small_rect.end.x, small_rect.start.y:small_rect.end.y, :]
		return small_dict

def get_tile_list():
	tiles = []
	with open(pytiles_path, 'r') as f:
		for json_tile in json.load(f):
			tile = geom.Point(int(json_tile['x']), int(json_tile['y']))
			tile.region = json_tile['region']
			tiles.append(tile)
	return tiles

def get_starting_locations(gcs, segment_length, region=None):
	all_starting_locations = {}
	with open(startlocs_path, 'r') as f:
		# top-level is dict from tile to starting location lists
		for tile, locs in json.load(f).items():
			tile_region = tile.split('_')[0]

			if region is not None and tile_region != region:
				continue
			elif tile_region not in gcs:
				continue

			starting_locations = []
			# each loc is a dict with keys 'x', 'y', 'edge_id'
			for loc in locs:
				point = geom.Point(int(loc['x']), int(loc['y']))
				edge_pos = gcs[tile_region].graph.edges[int(loc['edge_id'])].closest_pos(point)
				next_positions = graph.follow_graph(edge_pos, segment_length)

				if not next_positions:
					continue

				starting_locations.append([{
					'point': point,
					'edge_pos': edge_pos,
				}, {
					'point': next_positions[0].point(),
					'edge_pos': next_positions[0],
				}])
			all_starting_locations[tile] = starting_locations
	return all_starting_locations

class Tiles(object):
	def __init__(self, paths_per_tile_axis, segment_length, parallel_tiles, tile_mode):
		self.search_rect_size = tile_size / paths_per_tile_axis
		self.segment_length = segment_length
		self.parallel_tiles = parallel_tiles
		self.tile_mode = tile_mode

		# load tile list
		# this is a list of point dicts (a point dict has keys 'x', 'y')
		# don't include test tiles
		print 'reading tiles'
		self.all_tiles = get_tile_list()
		self.cache = TileCache(limit=self.parallel_tiles, mode=self.tile_mode)

		self.gcs = {}

	def get_gc(self, region):
		if region in self.gcs:
			return self.gcs[region]
		fname = os.path.join(graph_dir, region + '.graph')
		g = graph.read_graph(fname)
		gc = graph.GraphContainer(g)
		self.gcs[region] = gc
		return gc

	def cache_gcs(self, regions):
		for region in regions:
			print 'reading graph for region {}'.format(region)
			self.get_gc(region)

	def prepare_training(self):
		self.cache_gcs(REGIONS)
		print 'get starting locations'
		self.all_starting_locations = get_starting_locations(self.gcs, self.segment_length)

		def tile_filter(tile):
			if tile.region not in REGIONS:
				return False
			rect = geom.Rectangle(
				tile.scale(tile_size),
				tile.add(geom.Point(1, 1)).scale(tile_size)
			)
			starting_locations = self.all_starting_locations['{}_{}_{}'.format(tile.region, tile.x, tile.y)]
			starting_locations = [loc for loc in starting_locations if rect.add_tol(-window_size).contains(loc[0]['point'])]
			return len(starting_locations) > 0
		self.train_tiles = filter(tile_filter, self.all_tiles)

		old_len = len(self.train_tiles)
		self.train_tiles = [tile for tile in self.train_tiles if tile.region in TRAINING_REGIONS]
		print 'go from {} to {} tiles after excluding regions'.format(old_len, len(self.train_tiles))
		random.shuffle(self.train_tiles)

	def get_tile_data(self, region, rect):
		gc = self.get_gc(region)
		midpoint = rect.start.add(rect.end.sub(rect.start).scale(0.5))
		x = midpoint.x / tile_size
		y = midpoint.y / tile_size
		k = '{}_{}_{}'.format(region, x, y)
		starting_locations = get_starting_locations(self.gcs, self.segment_length, region=region)[k]
		starting_locations = [loc for loc in starting_locations if rect.add_tol(-window_size).contains(loc[0]['point'])]
		return {
			'region': region,
			'rect': rect,
			'search_rect': rect.add_tol(-window_size/2),
			'cache': self.cache,
			'starting_locations': starting_locations,
			'gc': gc,
		}

	def get_test_tile_data(self):
		if 'chicago' not in REGIONS:
			return None

		rect = geom.Rectangle(
			geom.Point(1024, -8192),
			geom.Point(4096, -5376),
		)

		starting_locations = self.all_starting_locations['chicago_0_-2']
		starting_locations = [loc for loc in starting_locations if rect.add_tol(-window_size).contains(loc[0]['point'])]
		return {
			'region': 'chicago',
			'rect': rect,
			'search_rect': rect.add_tol(-window_size/2),
			'cache': self.cache,
			'starting_locations': starting_locations,
			'gc': self.gcs['chicago'],
		}

	def num_tiles(self):
		return len(self.train_tiles)

	def num_input_channels(self):
		return 5

	def get_training_tile_data(self, tile_idx):
		tile = self.train_tiles[tile_idx]
		return self.get_training_tile_data_normal(tile)

	def get_training_tile_data_normal(self, tile, tries=0):
		rect = geom.Rectangle(
			tile.scale(tile_size),
			tile.add(geom.Point(1, 1)).scale(tile_size)
		)

		if tries < 3:
			search_rect_x = random.randint(window_size/2, tile_size - window_size/2 - self.search_rect_size)
			search_rect_y = random.randint(window_size/2, tile_size - window_size/2 - self.search_rect_size)
			search_rect = geom.Rectangle(
				rect.start.add(geom.Point(search_rect_x, search_rect_y)),
				rect.start.add(geom.Point(search_rect_x, search_rect_y)).add(geom.Point(self.search_rect_size, self.search_rect_size)),
			)
			starting_locations = self.all_starting_locations['{}_{}_{}'.format(tile.region, tile.x, tile.y)]
			starting_locations = [loc for loc in starting_locations if search_rect.add_tol(-window_size/4).contains(loc[0]['point'])]
		else:
			starting_locations = self.all_starting_locations['{}_{}_{}'.format(tile.region, tile.x, tile.y)]
			starting_locations = [loc for loc in starting_locations if rect.add_tol(-window_size).contains(loc[0]['point'])]
			starting_location = random.choice(starting_locations)
			search_rect_min = starting_location[0]['point'].sub(geom.Point(self.search_rect_size / 2, self.search_rect_size / 2)).sub(rect.start)

			if search_rect_min.x < window_size/2:
				search_rect_min.x = window_size/2
			elif search_rect_min.x > tile_size - window_size/2 - self.search_rect_size:
				search_rect_min.x = tile_size - window_size/2 - self.search_rect_size

			if search_rect_min.y < window_size/2:
				search_rect_min.y = window_size/2
			elif search_rect_min.y > tile_size - window_size/2 - self.search_rect_size:
				search_rect_min.y = tile_size - window_size/2 - self.search_rect_size

			search_rect = geom.Rectangle(
				rect.start.add(search_rect_min),
				rect.start.add(search_rect_min).add(geom.Point(self.search_rect_size, self.search_rect_size)),
			)
			starting_locations = [starting_location]

		if not starting_locations:
			return self.get_training_tile_data_normal(tile, tries + 1)
		return {
			'region': tile.region,
			'rect': rect,
			'search_rect': search_rect,
			'cache': self.cache,
			'starting_locations': starting_locations,
			'gc': self.gcs[tile.region],
		}
