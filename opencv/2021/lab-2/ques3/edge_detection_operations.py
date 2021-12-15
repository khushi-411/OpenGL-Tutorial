# ------------------------------------------------------
# Edge Detection Operations
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
img = cv2.imread(img_path, 0)

# Kernel
kernel = np.ones((5, 5), np.uint8)

# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)

# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Along X-Axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Along Y-Axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Along XY-Axis

# Prewitt: the x, and y-derivative filters are weighted with the standard averaging filter
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
prewittx = cv2.filter2D(img_blur, -1, kernelx)
prewitty = cv2.filter2D(img_blur, -1, kernely)
prewittxy = prewittx + prewitty

# Canny
canny = cv2.Canny(img, threshold1=100, threshold2=200)

# LoG Detector: Laplacian of Gaussian
log_detector = cv2.Laplacian(img_gray, ddepth=cv2.CV_16S, ksize=5)

# To display Image
#cv2.imshow('Sobel X', sobelx)
#cv2.imshow('Sobel Y', sobely)
#cv2.imshow('Sobel XY', sobelxy)
#cv2.imshow('Prewitt X', prewittx)
#cv2.imshow('Prewitt Y', prewitty)
#cv2.imshow('Prewitt XY' prewittxy)
#cv2.imshow('Canny', canny)
#cv2.imshow('Log Detector', log_detector)

# To save image
cv2.imwrite('../images/output_images/ques3_sobelx.png', sobelx)
cv2.imwrite('../images/output_images/ques3_sobely.png', sobely)
cv2.imwrite('../images/output_images/ques3_sobelxy.png', sobelxy)
cv2.imwrite('../images/output_images/ques3_prewittx.png', prewittx)
cv2.imwrite('../images/output_images/ques3_prewitty.png', prewitty)
cv2.imwrite('../images/output_images/ques3_canny.png', canny)
cv2.imwrite('../images/output_images/ques3_log_detector.png', log_detector)

# Waits till any key is pressed
cv2.waitKey(0)

# Closing all open windows
cv2.destroyAllWindows()
