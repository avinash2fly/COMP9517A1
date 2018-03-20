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




def otsu_threshold(img):
    fn_min = np.inf
    thresh = -1
    new_image = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    hist_norm = hist.ravel()/hist.max()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    for i in range(1, 256):
        p1, p2 = np.hsplit(hist_norm, [i])  # probabilities
        q1, q2 = Q[i], Q[255] - Q[i]  # cum sum of classes
        if q1 == 0:
            q1 = 0.00000001
        if q2 == 0:
            q2 = 0.00000001
        b1, b2 = np.hsplit(bins, [i])  # weights
        # finding means and variances
        m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
        v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2
        # calculates the minimization function
        fn = v1 * q1 + v2 * q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
            if img[i][j] >= thresh:
                new_image[i][j] = 255
    return new_image,thresh

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example with long option names')

    parser.add_argument('--input', action="store",nargs='+')
    parser.add_argument('--output', action="store")
    # parser.add_argument('--threshold', action='store_true', default=False)
    results = parser.parse_args()

    if not results.input or len(results.input)<2:
        print('Required input argumenst : input_image grid_size')
        exit(1)
    if not results.output:
        print('Required ouput image path')
        exit(1)

    print(results)
    # exit(0)
    img = cv.imread(results.input[0],0)
    blur = cv.GaussianBlur(img,(5,5),0)
    img=blur
    height = img.shape[0]
    width = img.shape[1]
    grid = int(results.input[1])
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    finalData = np.zeros((0, 0), dtype=img.dtype)
    for y in range(0,height,grid):
        tempData=np.zeros((0, 0), dtype=img.dtype)
        for x in range(0,width,grid):
            roi_gray = img[y:y + grid, x:x + grid]
            new_image,thres = otsu_threshold(roi_gray)
            if tempData.shape[0] is 0:
                tempData = new_image
            else:
                tempData = np.concatenate((tempData,new_image),axis=1)
            print(thres,roi_gray.shape)

        if finalData.shape[0] is 0:
            finalData = tempData
        else:
            finalData = np.concatenate((finalData,tempData),axis=0)
            # cv.imwrite("image.jpg", roi_gray)
    print(finalData.shape)
    cv.imwrite(results.output, finalData)