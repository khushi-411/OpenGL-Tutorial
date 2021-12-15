###############################################
# 
# Created by Khushi Agrawal on 05/10/21.
# Copyright (c) 2021 Khushi Agrawal. All rights reserved.
#
# Lab Program 1
# Contents: Perform grey-level transformation
#   - Negative Image Transformation
#   - Logarithmic Image Transformation
#   - Power Law Transformation
#
##############################################

import cv2
import numpy as np
import math 

# Negative image tranformation
def neg_trans(img):
	height = img.shape[0]
	width = img.shape[1]
	for i in range(height):
		for j in range(width):
			img[i][j] = 255 - int(img[i][j])

	return img

# Logarithmic image transformation
def log_trans(img):
	height = img.shape[0]
	width = img.shape[1]
	val = img.flatten()
	maxv = max(val)
	c = 255/(math.log(1+maxv))
	for i in range(height):
		for j in range(width):
			new_val = int(c * (math.log( 1 + int(img[i][j]))))
			if new_val > 255:
				new_val = 255
			img[i][j] = new_val

	return img

# Power Law Transformation
def powerLawTransform(c,pixel,gamma):
    new_pixel = c*pow(pixel,(1/gamma))
    return round(new_pixel)

def power_law(img, c, gamma):
    height = img.shape[0]
    width = img.shape[1]
    for w in range(width):
        for h in range(height):
            #retrieve the pixel value
            p = int(img[w][h])
            #add value to the pixel
            img[w,h] = powerLawTransform(c,p,gamma)
    return img

# ERROR (Solved): https://stackoverflow.com/questions/52676020/opencv-src-empty-in-function-cvtcolor-error
img = 'images/Aaron_Eckhart_0001.jpg'
image = cv2.imread(img)
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('images/ques_1_gray_image.png', gray_img)

neg_img = neg_trans(gray_img)
cv2.imwrite('images/ques_1_neg_img.png',neg_img)
print("Negative transformed image is saved as ques_1_neg_img.png.")

log_img = log_trans(gray_img)
cv2.imwrite('images/ques_1_log_img.png',log_img)
print("Log-transformation output image is saved as ques_1_log_img.png.")

#The scaling constant c is chosen so that the maximum output value is 255
c = 255/(math.log10(1+ np.amax(gray_img)))
gamma = float(input("Enter a value for gamma : "))

pow_law = power_law(gray_img, c, gamma)
cv2.imwrite('images/ques_1_power_law_img_(y-9).png', pow_law)
print("Power Law Transformation is saved as ques_1_power_law_img.png")
