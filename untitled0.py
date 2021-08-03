#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 13:14:25 2021

@author: allen
"""

mapFinalScaled = np.round(255 * mapFinal).astype(int)
print("here")
# Set up the detector with default parameters.
detector = cv.SimpleBlobDetector_create()
print("here")
# Detect blobs.
keypoints = detector.detect(mapFinalScaled)
print("here")
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv.drawKeypoints(mapFinal, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# Show keypoints
print("here")
plt.imshow(im_with_keypoints)
print("here")