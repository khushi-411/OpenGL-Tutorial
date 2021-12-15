# ------------------------------------------------------
# Edge Detection Operations: Sobel Edge Detector
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
kernel = np.ones((5, 5), np.uint8)

# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)

# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Along X-Axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Along Y-Axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Along XY-Axis


# To display Image
#cv2.imshow('Sobel X', sobelx)
#cv2.imshow('Sobel Y', sobely)
#cv2.imshow('Sobel XY', sobelxy)

# To save image
cv2.imwrite('../images/output_images/ques3_sobelx.png', sobelx)
cv2.imwrite('../images/output_images/ques3_sobely.png', sobely)
cv2.imwrite('../images/output_images/ques3_sobelxy.png', sobelxy)

# Waits till any key is pressed
cv2.waitKey(0)

# Closing all open windows
cv2.destroyAllWindows()
