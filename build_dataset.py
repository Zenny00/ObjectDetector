from re import X
from utilFunctions.IntersectionOverUnion import compute_intersection_over_union
from utilFunctions import config
from bs4 import BeautifulSoup
from imutils import paths
import cv2
import os

for dir_path in (config.PATH_POSITIVE, config.PATH_NEGATIVE):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

image_paths = list(paths.list_images(config.PATH_OYSTERS))

totalPositive = 0
totalNegative = 0

for (i, imagePath) in enumerate(image_paths):
    print("[INFO] processing image {}/{}...".format(i+1, len(image_paths)))

    filename = imagePath.split(os.path.sep)[-1]
    filename = filename[:filename.rfind(".")]
    annotationPath = os.path.sep.join([config.PATH_OYSTERS_ANNOTATIONS, "{}.xml".format(filename)])

    contents = open(annotationPath).read()
    soup = BeautifulSoup(contents, "html.parser")
    groundTruthBoxes = []

    width = int(soup.find("width").string)
    height = int(soup.find("height").string)

    for o in soup.find_all("object"):
        label = o.find("name").string
        xMin = int(o.find("xmin").string)
        yMin = int(o.find("ymin").string)
        xMax = int(o.find("xmax").string)
        yMax = int(o.find("ymax").string)

        xMin = max(0, xMin)
        yMin = max(0, yMin)
        xMax = max(0, xMax)
        yMax = max(0, yMax)

        groundTruthBoxes.append((xMin, yMin, xMax, yMax))

    image = cv2.imread(imagePath)

    selectiveSearch = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    selectiveSearch.setBaseImage(image)
    selectiveSearch.switchToSelectiveSearchFast()
    rectangles = selectiveSearch.process()
    proposedRectangles = []

    for (x, y, w, h) in rectangles:
        proposedRectangles.append((x, y, x+w, y+h))
        
    positiveROIs = 0
    negativeROIs = 0

    for proposedRectangle in proposedRectangles[:config.MAX_PROPOSALS]:
        (propStartX, propStartY, propEndX, propEndY) = proposedRectangle

        for groundTruthBox in groundTruthBoxes:
            intersection_over_union = compute_intersection_over_union(groundTruthBox, proposedRectangle)
            (groundTruthStartX, groundTruthStartY, groundTruthEndX, groundTruthEndY) = groundTruthBox
            roi = None
            outputPath = None

            if intersection_over_union > 0.7 and positiveROIs <= config.MAX_POSITIVE:
                roi = image[propStartY:propEndY, propStartX:propEndX]
                filename = '{}.jpg'.format(totalPositive)
                outputPath = os.path.sep.join([config.PATH_POSITIVE, filename])

                positiveROIs += 1
                totalPositive += 1

            fullOverlap = propStartX >= groundTruthStartX
            fullOverlap = fullOverlap and propStartY >= groundTruthStartY
            fullOverlap = fullOverlap and propEndX <= groundTruthEndX
            fullOverlap = fullOverlap and propEndY <= groundTruthEndY

            if not fullOverlap and intersection_over_union < 0.05 and \
                negativeROIs <= config.MAX_NEGATIVE:

                roi = image[propStartY:propEndY, propStartX:propEndX]
                filename = "{}.jpg".format(totalNegative)
                outputPath = os.path.sep.join([config.PATH_NEGATIVE, filename])

                negativeROIs += 1
                totalNegative += 1

            if roi is not None and outputPath is not None:
                roi = cv2.resize(roi, config.INPUT_DIMS, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(outputPath, roi)

