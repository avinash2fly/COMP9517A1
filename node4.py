from filter import *


def getNeighbours(shape, i, j, k=4, allNeighbours=False):
    '''
    Returns coordinates of the neighbours as dictionary.

    shape			: The dimension of the image as a tuple (nrows, ncols)
    i 				: Current position x coordinates
    j 				: Current position y coordinates
    k 				: Deault is 4.
                      If k = 4 then only adjacent cells are considered as neighbours.
                      If k = 8 then all surrounding cells are considered as neighbours.
    allNeighbours	: Default is False. If True, al lsurrounding neighbours are
                      extracted. If False, only neighbours above and to the left
                      of the current position is considered.
    '''
    neighbours = {}
    if i > 0:
        neighbours['N'] = (i - 1, j)
    if j > 0:
        neighbours['W'] = (i, j - 1)

    if k == 4:
        return neighbours

    if i > 0 and j > 0:
        neighbours['NW'] = (i - 1, j - 1)
    if i > 0 and j < shape[1] - 1:
        neighbours['NE'] = (i - 1, j + 1)

    return neighbours


def countNodules(image, minSize, k=4, foreground=0):
    '''
    Returns a 2x2 array with labelled nodules.

    image 		: input image
    minSize		: minimum size of nodules
    k			: If k = 4 then only adjacent cells are considered as neighbours.
                  If k = 8 then all surrounding cells are considered as neighbours.
    foreground	: 0 if foreground is black, 1 if foreground is white.
    '''
    labels = {0: 0}  # Track labels equivalence
    labeledImg = np.zeros(image.shape, dtype=np.uint64)
    bgColour = 0 + 255 * (1 - foreground)

    # Convert image dimension so that a single cell only contains 1 value.
    if len(image.shape) > 2:
        image = image[:, :, 0]

    # First pass
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            # If cell is in background, continue to next cell
            if image[r, c] == bgColour:
                continue
            # Get a list of neighbours' coordinates
            neighbours = getNeighbours(image.shape, r, c, k=k)

            # Combine all neighbouring labels into a set
            labelGroup = set()
            for n in neighbours:
                if labeledImg[neighbours[n]] > 0:
                    labelGroup.add(labeledImg[neighbours[n]])

            # If none of the neighbours has label yet, assign a new label
            if len(labelGroup) == 0:
                labeledImg[r, c] = max(labels) + 1
                labelGroup.add(labeledImg[r, c])
            # Else, assign the minimum label from the neighbours.
            else:
                labeledImg[r, c] = min(labelGroup)

            # Combine all labels equivalence
            for label in labelGroup:
                if label not in labels:
                    continue
                labelGroup = labelGroup.union(labels[label])

            # Update labels equivalence
            for label in labelGroup:
                labels[label] = labelGroup

    # firstPassImg = labeledImg.copy()

    # Second pass
    for r in range(labeledImg.shape[0]):
        for c in range(labeledImg.shape[1]):
            # If cell is in background, continue to next cell
            if labeledImg[r, c] == 0:
                continue
            # Update label with the minimum equivalent label.
            labeledImg[r, c] = min(labels[labeledImg[r, c]])

    # Eliminated nodules which are smaller than minSize
    for label in np.unique(labeledImg):
        if labeledImg[labeledImg == label].size < minSize:
            labeledImg[labeledImg == label] = 0

    # secondPassImg = labeledImg.copy()

    # return firstPassImg, secondPassImg, labels
    return labeledImg


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


if __name__ == "__main__":

    # args = main(sys.argv)
    #
    # inputImage = args['input']
    #
    # try:
    #     n = int(args['gridSize'])
    # except:
    #     n = 4096
    #
    # try:
    #     minSize = int(args['size'])
    # except:
    #     minSize = 1
    #
    # # MANUAL INPUT IMAGE
    # # inputFile = './DataSamples/ductile_iron2-0.jpg'
    # # inputImage = cv.imread(inputFile, 0)
    # # minSize = 1
    #
    # # FROM WIKI
    # # inputImage = np.array([
    # # 	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    # # 	[0,0,255, 255,0,0,255,255,0,0,255,255,0,0,255,255,0],
    # # 	[0,255,255, 255,255,255,255,255,255,0,0,255,255,255,255,0,0],
    # # 	[0,0,0,255,255,255,255,0,0,0,255,255,255,255,0,0,0],
    # # 	[0,0,255,255,255,255,0,0,0,255,255,255,0,0,255,255,0],
    # # 	[0,255,255,255,0,0,255,255,0,0,0,255,255,255,0,0,0],
    # # 	[0,0,255,255,0,0,0,0,0,255,255,0,0,0,255,255,0],
    # # 	[0,0,0,0,0,0,255,255,255,255,0,0,255,255,255,255,0],
    # # 	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # # 	])
    #
    # # newImage = filteredOtsu(inputImage, n, grid=True)
    # newImage = filteredOtsu(inputImage, n, grid=True, prep=True, morphImg=True)
    newImage = cv.imread("binary.png", 0)
    # first, second, labels = countNodules(newImage, minSize, k=4, foreground=0)
    labeledImg = countNodules(newImage, 1, k=4, foreground=0)

    coloredImage = printNodules(labeledImg)
    cv.imwrite("binary_opencv.png", coloredImage)

    # if 'optional_output' in args:
    #     cv.imwrite(args['optional_output'], coloredImage)
    # elif 'output' in args:
    #     cv.imwrite(args['output'], coloredImage)

    print(len(np.unique(labeledImg)) - 1)







