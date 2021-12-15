# ------------------------------------------------------
# Morphological Operations: Dilation
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

# Dilation
dilation = cv2.dilate(img, kernel, iterations=1)

# To display Image
#cv2.imshow('Dilated Image', dilation)

# To save image
cv2.imwrite('../images/output_images/ques2_dilation.png', dilation)

# Waits till any key is pressed
cv2.waitKey(0)

# Closing all open windows
cv2.destroyAllWindows()
