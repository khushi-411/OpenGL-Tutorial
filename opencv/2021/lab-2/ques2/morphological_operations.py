# ------------------------------------------------------
# Morphological Operations
#
# Created by Khushi Agrawal on 19/09/21.
# Copyright (c) 2021 Khushi Agrawal. All rights reserved.
# 
# ------------------------------------------------------

import cv2
import numpy as np

# Image path
# Tried with other images to by changing the file names to: 
# Path for MSRA Images: ../images/MSRA-Images/ + IMG_0714.JPG, IMG_0753.JPG, IMG_0827.JPG, IMG_0870.JPG, IMG_2222.JPG
# Path for lwf Images: ../images/lwf/ + Emma_Watson_0005.jpg, Scott_Wolf_0001,jpg, Skip_Prosser_0001.jpg, Suzanne_Somers_0001.jpg, Tom_Cruise_0010.jpg
img_path = '../images/lwf/Emma_Watson_0005.jpg'

# Reading Image
img = cv2.imread(img_path, 0)

# Kernel
kernel = np.ones((5, 5), np.uint8)

# Erosion
erosion = cv2.erode(img, kernel, iterations=1)

# Dilation
dilation = cv2.dilate(img, kernel, iterations=1)

# Hit and miss
hit_miss = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)

# Tinning
#tinning = cv2.add(img, cv2.bitwise_not(hit_miss))
#tinning = img - hit_miss
tinning = cv2.bitwise_and(img, cv2.bitwise_not(hit_miss))

# Skeletonization
# Threshold the image
ret,img = cv2.threshold(img, 127, 255, 0)

# Step 1: Create an empty skeleton
size = np.size(img)
skel = np.zeros(img.shape, np.uint8)

# Get a Cross Shaped Kernel
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

# Repeat steps 2-4
while True:
    #Step 2: Open the image
    open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
    #Step 3: Substract open from the original image
    temp = cv2.subtract(img, open)
    #Step 4: Erode the original image and refine the skeleton
    eroded = cv2.erode(img, element)
    skel = cv2.bitwise_or(skel,temp)
    img = eroded.copy()
    # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
    if cv2.countNonZero(img)==0:
        break

# Thickening
thickening = cv2.bitwise_or(img, hit_miss)

# To display Image
#cv2.imshow('Eroded Image', erosion)
#cv2.imshow('Dilated Image', dilation)
#cv2.imshow('Hit And Miss', hit_miss)
#cv2.imshow('Tinning', tinning)
#cv2.imshow('Skeletonization', skel)
#cv2.imshow('Thickening', thickening)

# To save image
cv2.imwrite('../images/output_images/ques2_erosion.png', erosion)
cv2.imwrite('../images/output_images/ques2_dilation.png', dilation)
cv2.imwrite('../images/output_images/ques2_hit_miss.png', hit_miss)
cv2.imwrite('../images/output_images/ques2_tinning.png', tinning)
cv2.imwrite('../images/output_images/ques2_skeletonization.png', skel)
cv2.imwrite('../images/output_images/ques2_thickening.png', thickening)

# Waits till any key is pressed
cv2.waitKey(0)

# Closing all open windows
cv2.destroyAllWindows()
