##################################################################
# 
# Created by Khushi Agrawal on 01/10/21.
# Copyright (c) 2021 Khushi Agrawal. All rights reserved.
#
# Lab Program 4
# Contents: Perform SIFT (Scale-Invariant Feature Transform).
#
##################################################################

import cv2

#img = cv2.imread('images/MSRA-Images/IMG_0714.JPG')

img = cv2.imread('images/outputs/ques1_sift_1_gray.png')

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imwrite('images/outputs/ques1_sift_1_gray.png', gray_img)

sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray_img, None)

kp_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#cv2.imwrite('images/outputs/ques1_sift_1.png', kp_img)
cv2.imwrite('images/outputs/ques1_sift_1_output_.png', kp_img)
