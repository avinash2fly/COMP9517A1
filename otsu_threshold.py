from __future__ import print_function
import cv2 as cv
import numpy as np
from optparse import OptionParser
import os

# You shouldn't use any of the following OpenCV library functions:
# threshold
# adaptiveThreshold
# watershed
# findContours
# contourArea
# drawContours
# connectedComponents

import argparse

parser = argparse.ArgumentParser(description='Example with long option names')

#parser.add_argument("--input", type=int, help="the base")
parser.add_argument('--input', action="store")
parser.add_argument('--output', action="store")
parser.add_argument('--threshold', action='store_true', default=False)
results = parser.parse_args()

if not results.input:
    print('Required input image')
    exit(1)
if not results.output:
    print('Required ouput image path')
    exit(1)
# print(results)
# exit(0);
img = cv.imread(results.input,0)
print(img.shape)
#gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(img,(5,5),0)
# blur = cv.medianBlur(img,5)
img=blur
# find normalized_histogram, and its cumulative distribution function
hist = cv.calcHist([img],[0],None,[256],[0,256])
hist_norm = hist.ravel()/hist.max()
Q = hist_norm.cumsum()
bins = np.arange(256)
fn_min = np.inf
thresh = -1
#new_image = [img.shape[0]][img.shape[1]];
new_image = np.zeros((img.shape[0], img.shape[1]),dtype=img.dtype)

for i in range(1,256):
    p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
    q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
    if q1 == 0:
        q1 = 0.00000001
    if q2 == 0:
        q2 = 0.00000001
    b1,b2 = np.hsplit(bins,[i]) # weights
    # finding means and variances
    m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
    v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2
    # calculates the minimization function
    fn = v1*q1 + v2*q2
    if fn < fn_min:
        fn_min = fn
        thresh = i

for i in range(0,len(img)):
    for j in range(0,len(img[i])):
        if(img[i][j]>=thresh):
            new_image[i][j]=255;
        # else:
        #     new_image[i][j]=255;
cv.imwrite(results.output,new_image)
# cv.imshow('image',new_image)
# cv.waitKey(0)
# cv.destroyAllWindows()
if results.threshold:
    print(thresh)
