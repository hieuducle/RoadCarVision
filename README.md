# [PYTORCH] Counting the number and determine speed of cars, car brand classification
<p align="center">
 <h1 align="center">RoadCarVision</h1>
</p>

## Introduction
This is a project to automatically classify and check car speed, count the number of cars entering and exiting.

## Descriptions
* Use YOLO to locate the object and then enter it into the car brand classification model (this classification model has been built and developed by Kamwoh <a>https://github.com/kamwoh/Car-Model-Classification </a>).
* Estimate car speed based on object distance between frames
* Based on the location of the object, calculate the number of vehicles entering and exiting.
<p align="center">
  <img src="output/output.gif" width=600><br/>
  <i>Camera app demo</i>
</p>
