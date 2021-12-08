##################################################################
# 
# Created by Khushi Agrawal on 01/10/21.
# Copyright (c) 2021 Khushi Agrawal. All rights reserved.
#
# Lab Program 4
# Contents: Perform SIFT (Scale-Invariant Feature Transform), SURF (Speeded-Up Robust Features), BRIEF (Binary Robust Independent Elementary Features) and ORB (Oriented FAST and Rotated BRIEF).
#
##################################################################

import cv2

img = cv2.imread('images/MSRA-Images/IMG_0714.JPG')

# For gray-scale image
#img = cv2.imread('images/outputs/ques2_1_gray.png')

height, width = img.shape[:2]
center = (width/2, height/2)

# To tilt image.
rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=45, scale=1)
img = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(width, height))

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite('images/outputs/ques2_1_rot.png', img)
cv2.imwrite('images/outputs/ques2_1_gray.png', gray_img)

# -------------------------------------------------------------------------
# SURF

##########################################################################
#
# IMPORTANT: SURF IS DEPRICIATED IN OPENCV 4.5.x. We are getting the following error.
#
# Traceback (most recent call last):
#  File "/home/khushi/Documents/college/cv/191020429_Khushi-Agrawal_DSAI/ques1/surf_brief_orb.py", line 23, in <module>
#    surf = cv2.xfeatures2d.SURF_create()
# cv2.error: OpenCV(4.5.4-dev) /tmp/pip-req-build-iefu5nf2/opencv_contrib/modules/xfeatures2d/src/surf.cpp:1027: error: (-213:The function/feature is not implemented) This algorithm is patented and is excluded in this configuration; Set OPENCV_ENABLE_NONFREE CMake option and rebuild the library in function 'create'
#
###########################################################################

#surf = cv2.xfeatures2d.SURF_create()
#kp1, des1 = surf.detectAndCompute(gray_img, None)

#kp_img1 = cv2.drawKeypoints(img, kp1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#cv2.imwrite('images/outputs/ques1_surf_1.png', kp_img1)
#cv2.imwrite('images/outputs/ques1_surf_1_output_.png', kp_img1)

# --------------------------------------------------------------------------
# SIFT

sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray_img, None)

kp_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite('images/outputs/ques2_sift_1.png', kp_img)
#cv2.imwrite('images/outputs/ques2_sift_1_output_.png', kp_img)

# --------------------------------------------------------------------------
# BRIEF

star = cv2.xfeatures2d.StarDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

kp2 = star.detect(img, None)
kp2, des2 = brief.compute(img, kp2)

kp_img2 = cv2.drawKeypoints(img, kp2, None, color=(0, 255, 0), flags=0)

cv2.imwrite('images/outputs/ques2_brief_1.png', kp_img2)
#cv2.imwrite('images/outputs/ques2_brief_1_output_.png', kp_img2)

# --------------------------------------------------------------------------
# ORB

orb = cv2.ORB_create(nfeatures=2000)
kp3, des3 = orb.detectAndCompute(gray_img, None)

kp_img3 = cv2.drawKeypoints(img, kp3, None, color=(0, 255, 0), flags=0)

cv2.imwrite('images/outputs/ques2_orb_1.png', kp_img3)
#cv2.imwrite('images/outputs/ques2_orb_1_output_.png', kp_img3)
