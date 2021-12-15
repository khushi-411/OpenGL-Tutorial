##################################################################
# 
# Created by Khushi Agrawal on 10/11/21.
# Copyright (c) 2021 Khushi Agrawal. All rights reserved.
#
# Lab Program 5
# Contents: For an input image, tilt the image (between 5-20 degree) at all three axes but one axis at a time (this gives three resultant image, one each of x y and z respectively).
#
##################################################################

import cv2
import numpy as np

# Change image name to use diff image.
img = cv2.imread('images/IMG_2222.JPG')

height, width = img.shape[:2]
center = (width/2, height/2)

rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=15+90, scale=1)
img = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(width, height))

# Harris Corner Detection
def harris(img):

    #Converting to grayscale
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray_img)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    _img = np.copy(img)
    # Threshold for an optimal value, it may vary depending on the image.
    _img[dst>0.01*dst.max()]=[0,0,255]

    # to save the image
    cv2.imwrite('images/outputs/ques2_harris_3_y.png', _img)

# Shi-Tomasi Corner Detector
def shi_tomasi(img):

    # Converting to grayscale
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Specifying maximum number of corners as 1000
    # 0.01 is the minimum quality level below which the corners are rejected
    # 10 is the minimum euclidean distance between two corners
    corners_img = cv2.goodFeaturesToTrack(gray_img,1000,0.01,10)
    
    corners_img = np.int0(corners_img)
    
    __img = np.copy(img)
    
    for corners in corners_img:
       
        x,y = corners.ravel()
        # Circling the corners in green
        cv2.circle(__img,(x,y),3,[0,255,0],-1)

    cv2.imwrite('images/outputs/ques2_shi_tomasi_3_y.png', __img)

# FAST algorithm
def fast_algo(img):
    
    # Convert to gray-scale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    fast = cv2.FastFeatureDetector_create() 

    # Detect keypoints with non max suppression
    keypoints_with_nonmax = fast.detect(gray, None)

    # Disable nonmaxSuppression 
    fast.setNonmaxSuppression(False)

    # Detect keypoints without non max suppression
    keypoints_without_nonmax = fast.detect(gray, None)

    image_with_nonmax = np.copy(img)
    image_without_nonmax = np.copy(img)

    # Draw keypoints on top of the input image
    img1 = cv2.drawKeypoints(img, keypoints_with_nonmax, image_with_nonmax, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2 = cv2.drawKeypoints(img, keypoints_without_nonmax, image_without_nonmax, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imwrite('images/outputs/ques2_fast_3_with_y.png', img1)
    cv2.imwrite('images/outputs/ques2_fast_3_without_y.png', img2)

harris(img)
shi_tomasi(img)
fast_algo(img)
