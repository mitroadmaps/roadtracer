RoadTracer
==========

This is the code for ["RoadTracer: Automatic Extraction of Road Networks from Aerial Images"](https://roadmaps.csail.mit.edu/roadtracer/).

Code for the two baselines from the paper are in other repositories (but you'll also need the dataset preparation code in this repository):

* DeepRoadMapper: https://github.com/mitroadmaps/deeproadmapper/
* Our segmentation approach: https://github.com/mitroadmaps/roadcnn/

You will need gomapinfer (https://github.com/mitroadmaps/gomapinfer/) as a dependency.

The training/inference code is built on top of TensorFlow.


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


Training
--------

First, if you used different paths for the imagery/graphs/json, then update the paths at the top of `tileloader.py`.

Now we can train the model.

	mkdir /data/model/
	mkdir /data/model/model_latest
	mkdir /data/model/model_best
	python train.py /data/model/


Inference
---------

First, update `infer.py` (`MODEL_PATH`, `REGION`, and `TILE_START`/`TILE_END`).

Then:

	python infer.py /data/model/model_latest/model out.graph

This will output `out.graph`.
