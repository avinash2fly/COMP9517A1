from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse

# to get neighbours
def getAdjacentNodes(shape, i, j, k):


    neighbours = {}
    if i > 0:
        neighbours['N'] = (i - 1, j)
    if j > 0:
        neighbours['W'] = (i, j - 1)
    # four node
    if k == 4:
        return neighbours

    if i > 0 and j > 0:
        neighbours['NW'] = (i - 1, j - 1)
    if i > 0 and j < shape[1] - 1:
        neighbours['NE'] = (i - 1, j + 1)
    return neighbours


def twoPass(img, size, nodes, frontColor):

    labels = {0: 0}
    labeled_img = np.zeros(img.shape, dtype=np.uint64)
    bg_color = 0 + 255 * (1 - frontColor)

    if len(img.shape) > 2:
        img = img[:, :, 0]

    #first Pass
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == bg_color:
                continue
            neighboursList = getAdjacentNodes(img.shape, i, j, nodes)
            label_group = set()
            combineLabelsGroup(label_group, labeled_img, neighboursList)
            if len(label_group) == 0:
                labeled_img[i, j] = max(labels) + 1
                label_group.add(labeled_img[i, j])
            else:
                labeled_img[i, j] = min(label_group)
            label_group = MergeEquivalence(label_group, labels)
            for label in label_group:
                labels[label] = label_group

    #second pass
    for i in range(labeled_img.shape[0]):
        for j in range(labeled_img.shape[1]):
            if labeled_img[i, j] == 0:
                continue
            labeled_img[i, j] = min(labels[labeled_img[i, j]])

    for label in np.unique(labeled_img):
        if labeled_img[labeled_img == label].size < size:
            labeled_img[labeled_img == label] = 0

    return labeled_img


def MergeEquivalence(label_group, labels):
    for label in label_group:
        if label not in labels:
            continue
        label_group = label_group.union(labels[label])
    return label_group


def combineLabelsGroup(label_group, labeled_img, neighboursList):
    for n in neighboursList:
        if labeled_img[neighboursList[n]] > 0:
            label_group.add(labeled_img[neighboursList[n]])

def printNodules(image):

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
    parser = argparse.ArgumentParser(description='Example with long option names')
    parser.add_argument('--input', action="store")
    parser.add_argument('--size', action='store')
    parser.add_argument('--optional_output', action="store")

    results = parser.parse_args()
    grid = 0
    if not results.input:
        print('Required input argumenst : input_image')
        exit(1)
    if not results.size:
        print('Required grid size')
        grid = int(results.size)
        exit(1)

    img = cv.imread(results.input,0)
    labeledImg = twoPass(img, grid, 4, 0)
    print(len(np.unique(labeledImg)) - 1)
    coloredImage = printNodules(labeledImg)
    if results.optional_output:
        cv.imwrite(results.optional_output, coloredImage)



