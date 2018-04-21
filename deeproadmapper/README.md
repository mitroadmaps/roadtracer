DeepRoadMapper
==============

Training
--------

Suppose the imagery is in `/data/imagery` and the masks are in `/data/masks`. First, train the model:

	mkdir /data/deeproadmapper_model/
	mkdir /data/deeproadmapper_model/model_latest
	mkdir /data/deeproadmapper_model/model_best
	python train_segment.py /data/imagery/ /data/masks/ /data/deeproadmapper_model/

Next, run inference on the training regions:

	mkdir /data/deeproadmapper_train_out/
	mkdir /data/deeproadmapper_train_out/im
	mkdir /data/deeproadmapper_train_out/graph
	python infer_segment_training.py /data/imagery/ /data/masks/ /data/deeproadmapper_model/ /data/deeproadmapper_train_out/im/

Extract road network graphs for each training region, e.g.:

	python ../utils/mapextract.py /data/deeproadmapper_train_out/im/houston_0_0.png 10 /data/deeproadmapper_train_out/graph/houston_0_0.graph
	
Now we will use the segmentation output and inferred graphs from the training set to generate training examples for the connection classifier:

	mkdir /data/deeproadmapper_examples/
	mkdir /data/deeproadmapper_examples/good
	mkdir /data/deeproadmapper_examples/bad
	python label_training.py /data/imagery/ /data/graphs/ /data/deeproadmapper_train_out/im/ /data/deeproadmapper_train_out/graph/ /data/deeproadmapper_examples/

Then we can train the connection classifier:

	mkdir /data/deeproadmapper_cmodel/
	mkdir /data/deeproadmapper_cmodel/model_latest
	mkdir /data/deeproadmapper_cmodel/model_best
	python train_connect.py /data/deeproadmapper_examples/ /data/deeproadmapper_cmodel/

Inference
---------

First, run inference on the testing regions:

	mkdir /data/deeproadmapper_test_out/
	mkdir /data/deeproadmapper_test_out/im
	mkdir /data/deeproadmapper_test_out/graph
	python infer_segment_testing.py /data/imagery/ /data/deeproadmapper_model/ /data/deeproadmapper_test_out/im/

Extract road network graphs and run refinement steps, e.g.:

	python ../utils/mapextract.py /data/deeproadmapper_test_out/im/boston.png 10 /data/deeproadmapper_test_out/graph/boston.graph
	python clean.py /data/deeproadmapper_test_out/graph/boston.graph /data/deeproadmapper_test_out/graph/boston.clean.graph

Run the connection classifier to add missing connections, e.g.:

	python infer_connect.py /data/deeproadmapper_cmodel/ /data/testsat/boston.png /data/deeproadmapper_test_out/im/boston.png /data/deeproadmapper_test_out/graph/boston.clean.graph /data/deeproadmapper_test_out/graph/boston.connect.graph

Correct the coordinates of the graph vertices by adding an offset (which depends on the region). mapextract.py outputs coordinates corresponding to the imagery. However, the origin of the test image may not be at (0, 0), and fix.py accounts for this.

	python ../utils/fix.py boston /data/deeproadmapper_test_out/graph/boston.connect.graph /data/deeproadmapper_test_out/graph/boston.fix.graph
