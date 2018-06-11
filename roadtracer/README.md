RoadTracer
==========

Training
--------

First, if you used different paths for the imagery/graphs/json when preparing the dataset, then update the paths at the top of `tileloader.py`.

Now we can train the model.

	mkdir /data/model/
	mkdir /data/model/model_latest
	mkdir /data/model/model_best
	python train.py /data/model/ --t /data/imagery/ --g /data/graphs/ --j /data/json/


Inference
---------

Run `infer.py`:

	python infer.py /data/model/model_latest/model out.graph --t /data/imagery/ --g /data/graphs/ --j /data/json/

This will output `out.graph`.
