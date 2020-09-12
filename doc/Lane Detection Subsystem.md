# Lane Detection Subsystem
The lane detection subsystem processes the images from left camera to detect the lanes and calculate its radius of curvature.
![Alt text](images/Lane.png?raw=true "Title")
The workflow of this subsystem is divided into the following steps. 

## Pre-processing
The image is first standardised to 1280x720, then it is preprocessed by reducing noise and correcting the geometric distortion.

## Edge detection
In this step, an edge-detected binary image is generated.
First, a suitable edge-detection filter is applied to identify the edges. On which, different thresholds such as magnitude, direction and HSL are applied to minimize the potential lanes from the set of the edges detected.

## Perspective transform
To identify lane co-ordinates, a top view of the image is calculated by performing perspective transformation on the image using the following source and destination points.
source --> destination

## Histogram-based search
A histogram based search is applied to the combined binary image to identify the lane edges.
First, a histogram is generated from which two most prominent peaks are identified as bases of the lanes. Then, a sliding window is used to identify the lanes till the top of the image. Finally, reverse propective transformation is applied to convert the detected lane co-ordinates with respect to the initial image.

## Calculating the radius of curvature
The lane co-ordinates detected in the prospective transformed image are used to calculate the radius of curvature. Here, the lane co-ordinates are fit into a 2nd order polynomial equation to calculate the radius of curvature.  
<img src="https://render.githubusercontent.com/render/math?math=R_{curve}=\dfrac{[1+(\dfrac{dy}{dx})^2]^{3/2}}{|\dfrac{d^2x}{dy^2}|}" width="200">
