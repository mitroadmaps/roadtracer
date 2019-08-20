import geom

import math

ORIGIN_SHIFT = 2 * math.pi * 6378137 / 2.0

def lonLatToMeters(p):
	mx = p.x * ORIGIN_SHIFT / 180.0
	my = math.log(math.tan((90 + p.y) * math.pi / 360.0)) / (math.pi / 180.0)
	my = my * ORIGIN_SHIFT / 180.0
	return geom.FPoint(mx, my)

def metersToLonLat(p):
	lon = (p.x / ORIGIN_SHIFT) * 180.0
	lat = (p.y / ORIGIN_SHIFT) * 180.0
	lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180.0)) - math.pi / 2.0)
	return geom.FPoint(lon, lat)

def getMetersPerPixel(zoom):
	return 2 * math.pi * 6378137 / (2**zoom) / 256

def lonLatToPixel(p, origin, zoom):
	p = lonLatToMeters(p).sub(lonLatToMeters(origin))
	p = p.scale(1 / getMetersPerPixel(zoom))
	p = geom.FPoint(p.x, -p.y)
	p = p.add(geom.FPoint(256, 256))
	return p

def pixelToLonLat(p, origin, zoom):
	p = p.sub(geom.FPoint(256, 256))
	p = geom.FPoint(p.x, -p.y)
	p = p.scale(getMetersPerPixel(zoom))
	p = metersToLonLat(p.add(lonLatToMeters(origin)))
	return p

def lonLatToMapboxTile(p, zoom):
	n = 2**zoom
	xtile = int((p.x + 180.0) / 360 * n)
	ytile = int((1 - math.log(math.tan(p.y * math.pi / 180) + (1 / math.cos(p.y * math.pi / 180))) / math.pi) / 2 * n)
	return (xtile, ytile)

def lonLatToMapbox(p, zoom, origin_tile):
	n = 2**zoom
	x = (p.x + 180.0) / 360 * n
	y = (1 - math.log(math.tan(p.y * math.pi / 180) + (1 / math.cos(p.y * math.pi / 180))) / math.pi) / 2 * n
	xoff = x - origin_tile[0]
	yoff = y - origin_tile[1]
	return geom.FPoint(xoff, yoff).scale(256)

def mapboxToLonLat(p, zoom, origin_tile):
	n = 2**zoom
	x = p.x / 256.0 + origin_tile[0]
	y = p.y / 256.0 + origin_tile[1]
	x = x * 360.0 / n - 180
	y = math.atan(math.sinh(math.pi * (1 - 2.0 * y / n)))
	y = y * 180 / math.pi
	return geom.FPoint(x, y)
