---
toc: true
layout: post
description: VeloxMobo development process
categories: [learning]
comments: true
title:  "VeloxMobo - mobile robot development"
---
# VeloxMobo - mobile robot development

![]({{ site.baseurl }}/images/velox-dev/jetson.jpg "Jetson Nano with Stereo camera")

## Key Highlights

- Hobbyist project, **autonomous delivery** of light cargo (up to 2 kg) in Helsinki area.
- **Hardware**: Nvidia Jetson Nano microcomputer, stereo camera, monocular camera, LiDAR, GPS module.
- **Software**: lane and obstacle detection software (Python/C++) using OpenCV and Tensorflow. Navigation, decision making and path planning using Jetson OS and simultaneous mapping software.
- **Expected date of production**: May, 2021.

## Computer Vision and Mapping flow
<br>
![]({{ site.baseurl }}/images/velox-dev/planning.PNG "Sensing and planning")
<br>

## ML DevOps pipeline
<br>
![]({{ site.baseurl }}/images/velox-dev/devops-process.PNG "Continuous DevOps pipeline")
<br>

## Preliminary results
<br>
**City Center** <br>
![]({{ site.baseurl }}/images/velox-dev/results_street.avi "Object detection city center")
<br>
