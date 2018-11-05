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


Starting from Multiple Locations
--------------------------------

Using `infer.py` as in the previous section will start the iterative graph
construction process from one starting location taken from the ground truth map
dataset.

As we discuss in the paper, this may not work well if a city consists of
several weakly connected components. Instead, we can take starting locations
from another method, such as starting IGC from peaks in the segmentation
output.

One version of this is to first run RoadTracer with a very low threshold to
cover the entire city. Then, filter the output graph and only retain edges
where RoadTracer had a high confidence. Finally, restart the search with a
higher threshold, using the retained edges as starting locations.

To do this, use the `--f` and `--e` options:

	python infer.py /data/model/model_latest/model out-ny-0.03.graph --t /data/imagery/ --g /data/graphs/ --j /data/json/ --r 'new york' --s 0.03 --f 0.75
	python infer.py /data/model/model_latest/model out-ny-0.4.graph --t /data/imagery/ --g /data/graphs/ --j /data/json/ --r 'new york' --s 0.4 --e out-ny-0.03.graph

To generate precision-recall curve, 0.03 and 0.75 can be kept constant, while the second `--s` option (0.4 above) can be varied.
