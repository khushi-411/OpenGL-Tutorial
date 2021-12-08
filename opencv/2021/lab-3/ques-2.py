##################################################################
# 
# Created by Khushi Agrawal on 05/10/21.
# Copyright (c) 2021 Khushi Agrawal. All rights reserved.
#
# Lab Program 2
# Contents: Perform grey level slicing with and without background.
#
##################################################################

import cv2
import numpy as np

img = cv2.imread('images/Emma_Watson_0002.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
x = img.shape[0]
y = img.shape[1]
z = np.zeros((x, y))

# Gray Level Slicing with Background
for i in range(x):
    for j in range(y):
        if(img[i][j] > 50 and img[i][j] < 150):
            z[i][j] = 255
        else:
            z[i][j] = img[i][j]

equ1 = np.hstack((img, z))
cv2.imwrite('images/ques_2_gray_level_scaling_with_back.png', equ1)

# Gray Level Slicing without Background
img = cv2.imread('images/IMG_0758.JPG')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
x = img.shape[0]
y = img.shape[1]
z = np.zeros((x, y))

for i in range(x):
    for j in range(y):
        if(img[i][j] > 50 and img[i][j] < 150):
            z[i][j] = 255
        else:
            z[i][j] = 0

equ2 = np.hstack((img, z))
cv2.imwrite('images/ques_2_gray_level_scaling_without_back.png', equ2)
