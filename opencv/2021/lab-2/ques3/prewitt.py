# ------------------------------------------------------
# Edge Detection Operations: Prewitt Edge Detector
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
img_path = '../images/lwf/Scott_Wolf_0001.jpg'

# Reading Image
img = cv2.imread(img_path)

# Kernel
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)

# Prewitt: the x, and y-derivative filters are weighted with the standard averaging filter
prewittx = cv2.filter2D(img_blur, -1, kernelx)
prewitty = cv2.filter2D(img_blur, -1, kernely)
prewittxy = prewittx + prewitty

# To display Image
#cv2.imshow('Prewitt X', prewittx)
#cv2.imshow('Prewitt Y', prewitty)
#cv2.imshow('Prewitt XY' prewittxy)

# To save Image
cv2.imwrite('../images/output_images/ques3_prewittx.png', prewittx)
cv2.imwrite('../images/output_images/ques3_prewitty.png', prewitty)
cv2.imwrite('../images/output_images/ques3_prewittxy.png', prewittxy)
