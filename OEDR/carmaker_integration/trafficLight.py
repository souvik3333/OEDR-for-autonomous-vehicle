import cv2 

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import math


def standardize_input(image):
    
    standard_im = np.copy(image)
    standard_im = cv2.resize(standard_im,(32,32)) #Resizing image to size (32,32)
    rows = 4
    cols = 6
    i = standard_im.copy()
    #Cropping 4 rows from both upper and lower end of image 
    i = i[rows:-rows, cols:-cols, :]
    #Applying gaussian blur to image to remove noise
    i = cv2.GaussianBlur(i, (3, 3), 0)
    return i

def avg_value(rgb_image):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
#     h = hsv[:,:,0]
#     s = hsv[:,:,1]
#     v = hsv[:,:,2]

#     # Plot the original image and the three channels
#     f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
#     ax1.set_title('Standardized image')
#     ax1.imshow(test_im)
#     ax2.set_title('H channel')
#     ax2.imshow(h, cmap='gray')
#     ax3.set_title('S channel')
#     ax3.imshow(s, cmap='gray')
#     ax4.set_title('V channel')
#     ax4.imshow(v, cmap='gray')
    
    area = hsv.shape[0] * hsv.shape[1]
    #sum up the value to know color intensity 
    sum1 = np.sum(hsv[:, :, 2])
    #Find average color intensity of image
    avg1 = sum1 / area
    return int(avg1)

def create_feature(rgb_image):
    img = rgb_image.copy()
    #Create 3 slices of image vertically.
    upper_slice = img[0:7, :, :]
    middle_slice = img[8:15, :, :]
    lower_slice = img[16:24, :, :]
    #Find avergae value of each image.
    #To decide which traffic light might be on.
    u1 = avg_value(upper_slice)
    m1 = avg_value(middle_slice)
    l1 = avg_value(lower_slice)
    # print(u1,m1,l1)
    return u1,m1,l1

def estimate_label(rgb_image):
    u1,m1,l1 = create_feature(standardize_input(rgb_image))
 
    if(u1 > m1 and u1 > l1):
        return [1,0,0]
    elif(m1 > l1):
        return [0,1,0]
    else:
        return [0,0,1]