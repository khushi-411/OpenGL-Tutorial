# ------------------------------------------------------
# Edge Detection Operations: LoG Detector
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

# LoG Detector: Laplacian of Gaussian
log_detector = cv2.Laplacian(img_gray, ddepth=cv2.CV_16S, ksize=5)

# To display image
#cv2.imshow('Log Detector', log_detector)

# To save image
cv2.imwrite('../images/output_images/ques3_log_detector.png', log_detector)

# Waits till any key is pressed
cv2.waitKey(0)

# Closing all open windows
cv2.destroyAllWindows()
