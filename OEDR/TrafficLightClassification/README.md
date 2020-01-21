# Traffic Light Classification Module

## About
This module is dedicated to classifying the detected traffic lights into red, yellow and green.

## Architecture
The input, from the YOLO module detecting the traffic light, is processed through the pipeline with the following steps:
1. Image Pre-processing
2. Feature Extraction
3. Prediction

(image)

## Documentation
### Image Pre-processing
The following steps are involved in the image pre-processing:
1. The traffic light module needs the input image to be of constant size. So, first the image will be normalised to a 32x32 image.
2. Next, we crop 4 rows from both upper and lower end of the image as the whole traffic light board is not needed.
3. Finally, the unnecessary information and noise in the 32x32 image will be removed with the help of a filter. In this we have used a Gaussian Filter which is a linear filter.

### Feature Extraction
The brightness feature is used to detect illuminated section of the image.
1. First, the RGB image is converted to HSV image.
1. Then, the image is divided into three segments (top, middle and bottom) for the corresponding traffic lights.
2. Finally, we calculate the average brightness values of each segment.

### Prediction
This step compares the average brightness of the three segments and predicts the light corresponding to the highest brightness as the illuminated one.

## Dependencies
Python 3.6.8
