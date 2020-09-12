# Traffic Sign Classification Module
![Alt text](images/Traffic_Sign.png?raw=true "Title")

The input, from the YOLO module detecting the traffic signs, is processed through the pipeline with the following steps:

## Image Pre-processing
The following steps are involved in the image pre-processing:
1. RGB input image is converted into grey scale to reduce complexity (reducing 3 RGB chanels to 1 grey scale).
2. The traffic sign module needs the input image to be of constant size. So, the image will be normalised to a 32x32 image.

## Detection Module
 The architecture consists of three convolutional layers, two average pooling layers, followed by a flattening convolutional layer, then two fully-connected layers and finally a softmax classifier.
