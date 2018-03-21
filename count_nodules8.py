from __future__ import print_function
import cv2 as cv
import numpy as np
import math
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

equival ={}
labels=set()
def checkData(img,i,j):
    if i < 0 or j < 0:
        return 0
    else:
        return img[i][j]

def getMinimumValue(value):
    if value in equival:
        return getMinimumValue(equival[value])
    else:
        return value

def getNeighbourValue(img, i, j,counter):
    value =  img[i][j]
    if checkData(img,i - 1,j) and checkData(img,i,j - 1):
        value =  min(img[i - 1][j],img[i][j - 1])
        equival[max(img[i - 1][j],img[i][j - 1])] = getMinimumValue(value)
    elif checkData(img,i-1,j):
        value = img[i-1][j]
    elif checkData(img,i,j - 1):
        value = img[i][j - 1]
    else:
        counter = counter + 1
        value =  counter
    return counter,value


def twoPass(img):
    pass1_image = img
    row = img.shape[0]
    col = img.shape[1]
    counter=0;
    for i in range(0,row):
        for j in range(0,col):
            if int(img[i][j]) is not 0:
                counter,value  = getNeighbourValue(img, i, j, counter)
                pass1_image[i][j] = value
    print()
    print(pass1_image)

    for i in range(0,row):
        for j in range(0,col):
            if (int(img[i][j]) is not 0) and (img[i][j] in equival):
                if pass1_image[i][j] == equival[pass1_image[i][j]]:
                    labels.add(pass1_image[i][j])
                pass1_image[i][j] = equival[pass1_image[i][j]]
            elif int(pass1_image[i][j]) is not 0:
                labels.add(pass1_image[i][j])
    print()
    print(pass1_image)
    #return new_image,thresh

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Example with long option names')
    #
    # parser.add_argument('--input', action="store")
    # parser.add_argument('--size', action='store')
    # parser.add_argument('--optional_output', action="store")
    #
    # results = parser.parse_args()
    # grid = 0
    # if not results.input:
    #     print('Required input argumenst : input_image')
    #     exit(1)
    # if not results.size:
    #     print('Required grid size')
    #     grid = int(results.size)
    #     exit(1)
    #
    # print(results)
    # # exit(0)
    # img = cv.imread(results.input,0)
    r = np.random.RandomState(1234)
    a = r.rand(5,10)
    img = np.zeros([5,10],dtype=int)
    for i in range(0,5):
        for j in range(0,10):
            if a[i][j] >= 0.5:
                img[i][j] = 1
    print(img)
    twoPass(img)
    print()
    print(equival)
    print()
    print(labels)
    #cv.imwrite(results.output, finalData)