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
tempEquival ={}
equival ={}
labels=set()
def checkData(img,i,j):
    if i < 0 or j < 0:
        return 0
    else:
        return img[i][j]

def getMinimumValue(value):
    if value in tempEquival:
        if value == min(tempEquival[value]):
            return value
        else:
            return getMinimumValue(min(tempEquival[value]))
    else:
        return value


# def getNeighbourValue(img, i, j,counter):
#     value =  img[i][j]
#     if checkData(img,i-1,j):
#         value = img[i-1][j]
#         equival[max(img[i - 1][j], img[i][j - 1])] = value
#
#     if checkData(img,i,j - 1):
#         value = img[i][j - 1]
#
#     else:
#         counter = counter + 1
#         value =  counter
#     return counter,value


def getNeighbourValue(img, i, j,counter):
    value =  img[i][j]
    if checkData(img,i - 1,j) and checkData(img,i,j - 1):
        value =  min(img[i - 1][j],img[i][j - 1])

    elif checkData(img,i-1,j):
        value = img[i-1][j]

    elif checkData(img,i,j - 1):
        value = img[i][j - 1]

    else:
        counter = counter + 1
        value =  counter

    t_key = max(checkData(img,i-1,j), checkData(img,i,j - 1))
    if t_key is not 0:
        if t_key in tempEquival:
            temp = tempEquival[t_key]
            tempEquival[t_key].add(value)
        else:
            tempEquival[t_key] = set({value})
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

    for key in tempEquival:
        equival[key] = getMinimumValue(min(tempEquival[key]))

    for i in range(0,row):
        for j in range(0,col):
            if (int(img[i][j]) is not 0) and (img[i][j] in equival):
                # if pass1_image[i][j] == equival[pass1_image[i][j]]:
                labels.add(pass1_image[i][j])
                pass1_image[i][j] = equival[pass1_image[i][j]]
            elif int(pass1_image[i][j]) is not 0:
                labels.add(pass1_image[i][j])
    print()
    print(pass1_image)
    #return new_image,thresh


def printNodules(image):
    '''
    Return an image containing nodules printed in different colour.
    '''

    B = [75, 75, 25, 200, 48, 180, 240, 230, 60, 190, 128, 255, 40, 200, 0, 195, 0, 180, 128, 128, 255]
    G = [25, 180, 225, 130, 130, 30, 240, 50, 245, 190, 128, 190, 110, 250, 0, 255, 128, 215, 0, 128, 255]
    R = [230, 60, 255, 0, 245, 145, 70, 240, 210, 250, 0, 230, 170, 255, 128, 170, 128, 255, 0, 128, 255]

    colors = np.stack((B, G, R), axis=-1)
    colormap = {}
    counter = 0

    for label in np.unique(image):
        if label == 0:
            colormap[label] = np.array([0, 0, 0])
        else:
            colormap[label] = colors[counter % colors.shape[0]]
        counter += 1

    shape = image.shape
    coloredImage = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            coloredImage[r, c, :] = colormap[image[r, c]]

    return coloredImage


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
    img = cv.imread("binary.png",0)
    #
    # img = cv.imread("DataSamples/ductile_iron2-0.jpg", 0)
    # (T, thresh) = cv.threshold(img, 146, 255, cv.THRESH_BINARY)
    # r = np.random.RandomState(1234)
    # a = r.rand(5,10)
    # img = np.zeros([5,10],dtype=int)
    # for i in range(0,5):
    #     for j in range(0,10):
    #         if a[i][j] >= 0.5:
    #             img[i][j] = 1
    img = cv.bitwise_not(img)
    # img[img > 0] = 1
    print(img)
    # img = img[100:300, 200:400]
    twoPass(img)
    print()
    print(equival)
    print()
    print(labels)
    # img = cv.bitwise_not(img)
    # backtorgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    # img = cv.bitwise_not(img)
    # ret, labels = cv.connectedComponents(thresh)

    # label_hue = np.uint8(179 * img / np.max(img))
    # blank_ch = 255 * np.ones_like(label_hue)
    # labeled_img = cv.merge([label_hue, blank_ch, blank_ch])
    # # cvt to BGR for display
    # labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)
    #
    # # set bg label to black
    # labeled_img[label_hue == 0] = 255

    # output = cv.connectedComponentsWithStats(thresh, 4, cv.CV_32S)
    # img = output[1]
    # img = cv.bitwise_not(img)

    # backtorgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    # print(output)
    img = printNodules(img)
    cv.imwrite("binary_opencv.png", printNodules(img))