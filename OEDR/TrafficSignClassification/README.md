# Traffic Sign Classification Module

## About
This module is dedicated to classifying the detected traffic lights into red, yellow and green.

## Architecture
The input, from the YOLO module detecting the traffic signs, is processed through the pipeline with the following steps:
1. Image Pre-processing
2. Detection Module
(image)

## Documentation
### Image Pre-processing
The following steps are involved in the image pre-processing:
1. RGB input image is converted into grey scale to reduce complexity (reducing 3 RGB chanels to 1 grey scale).
2. The traffic sign module needs the input image to be of constant size. So, the image will be normalised to a 32x32 image.

### Detection Module
 The architecture consists of three convolutional layers, two average pooling layers, followed by a flattening convolutional layer, then two fully-connected layers and finally a softmax classifier.
## Dependencies
Python 3.6.8  
Pytorch 1.3.0+cu100  
numpy 1.17.3  
matplotlib  3.1.1  
OpenCV 3.4.3  
pickle 3.0  
PIL 1.1.7  
