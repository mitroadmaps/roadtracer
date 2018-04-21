Our Segmentation Approach for RoadTracer Paper
==============================================

You will first need to follow dataset preparation instructions from RoadTracer code. You will also need to copy discoverlib folder from RoadTracer code.

Suppose the imagery is in `/data/imagery` and the masks are in `/data/masks`. First, train the model:

	mkdir /data/roadcnn_model/
	mkdir /data/roadcnn_model/model_latest
	mkdir /data/roadcnn_model/model_best
	python train.py /data/imagery/ /data/masks/ /data/roadcnn_model/

Then, you can run inference on the test regions:

	mkdir outputs
	python infer.py /data/imagery/ /data/roadcnn_model/

Extract graphs from the segmentation outputs:

	python ../utils/mapextract.py outputs/boston.png 50 outputs/boston.graph

Correct the coordinates of the graph vertices by adding an offset (which depends on the region). mapextract.py outputs coordinates corresponding to the imagery. However, the origin of the test image may not be at (0, 0), and fix.py accounts for this.

	python ../utils/fix.py boston outputs/boston.graph outputs/boston.fix.graph
