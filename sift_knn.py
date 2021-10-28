# John Strenio CS510 CV & DL Assignment1: Part1
# citations: https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html
#            https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
#            https://stackoverflow.com/questions/39263646/opencv-how-to-calculate-sift-descriptor-at-a-given-pixel

import numpy as np 
import cv2
from matplotlib import pyplot as plt

# read in images
img = cv2.imread('SIFT1_img.jpg')
img2 = cv2.imread('SIFT2_img.jpg')
img2 = cv2.resize(img2, (798, 1200))
img_h, img_w, img_chan = img.shape

# create and compute sift
sift = cv2.SIFT_create()
kp1, d1 = sift.detectAndCompute(img, None)
kp2, d2 = sift.detectAndCompute(img2, None)

# convert points to list for use
pts1 = cv2.KeyPoint_convert(kp1)
pts2 = cv2.KeyPoint_convert(kp2)

# for each keypoint find nearest corresponding keypoint
keypoint_matches = []
for i in range(len(d1)):
    distances = []
    for j in range(len(d2)):
        distances.append(np.linalg.norm(d1[i] - d2[j]))

    # store minimum distance in tuple with 2 corresponding keypoints
    min_dist = min(distances)
    idx_min = distances.index(min_dist)
    keypoint_matches.append(tuple((pts1[i], pts2[idx_min], min_dist)))

# sort and filter top 10% of keypoints
keypoint_matches.sort(key=lambda tup: tup[2])
top_kps = keypoint_matches[:int(len(keypoint_matches) * 0.05)]

# draw keypoints
img = cv2.drawKeypoints(img, kp1, img)
img2 = cv2.drawKeypoints(img2, kp2, img2)

# resize image and stitch
both = np.concatenate((img, img2), axis=1)

# recalculate image2's stiched keypoints positions and draw lines
img2_kps = []
print(top_kps[0])
print(top_kps[0][1])
for i in range(len(top_kps)):
    x, y = top_kps[i][1]
    img2_kps.append(tuple((int(x+img_w), int(y))))
    cv2.line(both, tuple(top_kps[i][0]), tuple(img2_kps[i]), (0, 255, 0), thickness=1)

# write image to file
cv2.imwrite('SIFT_stitch.jpg', both)
