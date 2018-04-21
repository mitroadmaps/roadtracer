Dataset Preparation
-------------------

The bounding boxes for the dataset regions are defined in `lib/regions.go`. There are 40 regions in all, with 25 used for training and 15 for testing.

First, obtain the satellite imagery from Google Maps. You will need an API key from https://developers.google.com/maps/documentation/static-maps/.

	mkdir /data/imagery/
	go run 1_sat.go APIKEY /data/imagery/

Next, download the OpenStreetMap dataset and extract crops of the road network graph from it. We convert the coordinates from longitude/latitude a coordinate system that matches the pixels from the imagery.

	wget https://planet.openstreetmap.org/pbf/planet-latest.osm.pbf -O /data/planet.osm.pbf
	mkdir /data/rawgraphs/
	go run 2_mkgraphs.go /data/planet.osm.pbf /data/rawgraphs/
	mkdir /data/graphs/
	go run 3_convertgraphs.go /data/rawgraphs/ /data/graphs/

Generate starting locations (used by RoadTracer for training):

	mkdir /data/json/
	go run 4_startlocs.go /data/graphs/
	mv pytiles.json starting_locations.json /data/json/

Generate road masks (used by baseline approaches for training):

	mkdir /data/masks/
	go run 5_truth_tiles.go /data/graphs/ /data/masks/

Create test satellite imagery files, which consist of four imagery tiles merged together:

	mkdir /data/testsat/
	python merge.py /data/imagery/ boston 1 -1 3 1 /data/testsat/boston.png
	python merge.py /data/imagery/ chicago -1 -2 1 0 /data/testsat/chicago.png
	python merge.py /data/imagery/ amsterdam -1 -1 1 1 /data/testsat/amsterdam.png
	python merge.py /data/imagery/ denver -1 -1 1 1 /data/testsat/denver.png
	python merge.py /data/imagery/ 'kansas city' -1 -1 1 1 '/data/testsat/kansas city.png'
	python merge.py /data/imagery/ la -1 -1 1 1 /data/testsat/la.png
	python merge.py /data/imagery/ montreal -1 -1 1 1 /data/testsat/montreal.png
	python merge.py /data/imagery/ 'new york' -1 -1 1 1 '/data/testsat/new york.png'
	python merge.py /data/imagery/ paris -1 -1 1 1 /data/testsat/paris.png
	python merge.py /data/imagery/ pittsburgh -1 -1 1 1 /data/testsat/pittsburgh.png
	python merge.py /data/imagery/ saltlakecity -1 -1 1 1 /data/testsat/saltlakecity.png
	python merge.py /data/imagery/ 'san diego' -1 -1 1 1 '/data/testsat/san diego.png'
	python merge.py /data/imagery/ tokyo -1 -1 1 1 /data/testsat/tokyo.png
	python merge.py /data/imagery/ toronto -1 -1 1 1 /data/testsat/toronto.png
	python merge.py /data/imagery/ vancouver -1 -1 1 1 /data/testsat/vancouver.png
