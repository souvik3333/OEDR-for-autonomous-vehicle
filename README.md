# OEDR Module for Autonomous Module using Simulators
## About
The key aspects of conventional autonomous driving such as cost of infrastructure and maintenance, the development of the autonomous vehicle will take a lot of time, effort and capital. By using simulators in the design and development of various modules of autonomous cars we can reduce this testing duration. In our project, we have employed the help of simulators to create a realistic imitation of the driving environment and developed an OEDR with stereo cameras. we have used IPG-CarMaker as our simulator for its real-time and efficient models.

## Architecture
This project is dedicated to developing an Object and Event Detection and Recognition (OEDR) module. The following is the architecture of the OEDR Module integrated with IPG-CarMaker.

![Alt text](doc/images/OEDR.png?raw=true "Title")

### IPG Movie
IPG-Movie, an application launched through IPG-CarMaker, streams images from the stereoscopic cameras through the specified port.
The images are classified into left and right based on the position of the cameras. Here, we are using the left camera as the primary camera.

### YOLOv3
This CNN based module is responsible for both static and dynamic object detection and classification. It takes in the left camera images from the IPG Movie as it is the primary camera of our module. The output of this module is further used by the following modules. 

### Lane Detection
Along with the YOLO module, the Lane detection module also uses the left frame from IPG-Movie. It employs Computer Vision techniques for detecting and calculating the radius of curvature of the lanes.

### Traffic Light Classifier
We have employed Computer Vision techniques for classifying the traffic lights. It takes the detected traffic light from the YOLO module as input.

### Traffic Sign Classifier
We have employed a Machine Learning model for classifying the traffic lights. It takes the detected traffic sign from the YOLO module as input.

### Depth Estimation
This module is for estimating the depth of different detected objects. As this module uses stereo depth estimation so, it takes the right camera image along with the output of the YOLO module as input.

### Augmentation
Finally, the outputs of all the modules are augmented to provide an augmented video output with all the detections and classifications available.

## Dependencies
The dependencies can be installed with "environment.yml" file. Detailed information can be found in <a href="./doc/getting_started.md">getting_started.md</a>
