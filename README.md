RoadTracer Code
===============

This is the code for ["RoadTracer: Automatic Extraction of Road Networks from Aerial Images"](https://roadmaps.csail.mit.edu/roadtracer/).

There are several components, and each folder has a README with more usage details:

* dataset: code for dataset preparation
* roadtracer: RoadTracer
* roadcnn: our segmentation approach (baseline)
* deeproadmapper: DeepRoadMapper (baseline)

You will need gomapinfer (https://github.com/mitroadmaps/gomapinfer/) as a dependency.

The training/inference code is built on top of TensorFlow.
