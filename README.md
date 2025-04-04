# [PYTORCH] Counting the number and determine speed of cars, car brand classification
<p align="center">
 <h1 align="center">RoadCarVision</h1>
</p>

## Introduction
This is a project to automatically classify and check car speed, count the number of cars entering and exiting.

## Descriptions
* Applied YOLOv8 for vehicle detection and bounding box extraction in traffic videos.
* Used a pre-trained car brand classification model from an external repository (this classification model has been built and developed by Kamwoh <a>https://github.com/kamwoh/Car-Model-Classification </a>). to identify vehicle brands
* Implemented logic to count incoming and outgoing vehicles in a two-way road scenario.

* Estimated vehicle speed based on movement across frames using object tracking and frame timestamps.

* Built the entire pipeline using Python and OpenCV, integrating detection, classification, tracking, and visualization.
</br>
<p align="center">
  <img src="output/output.gif" width=600><br/>
  <i>Demo</i>
</p>
