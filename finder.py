# import the necessary packages
import numpy as np
import argparse
import imutils
from matplotlib import pyplot as plt
import cv2

# Use command line arguments to read puzzles
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--puzzle", required = True,
    help = "Path to the puzzle image")
ap.add_argument("-w", "--waldo", required = True,
    help = "Path to the waldo image")
args = vars(ap.parse_args())

puzzle = cv2.imread('find_waldo.jpg')
img_gray = cv2.cvtColor(puzzle, cv2.COLOR_BGR2GRAY)
waldo = cv2.imread('waldo.png',0)

# store width and height
w, h = waldo.shape[::-1]
res = cv2.matchTemplate(img_gray,waldo,cv2.TM_CCOEFF_NORMED)
threshold = 0.6

# finding values exceeding threshold
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    #draw rectangle on places where it exceeds threshold
    cv2.rectangle(puzzle, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2)
cv2.imshow("Puzzle", imutils.resize(puzzle, height = 650))
cv2.waitKey(0)

# Offer to make Waldo more clear
user_input = input("Press 1 if your having trouble seeing Waldo \n")
user_int_put = int(user_input)

if user_int_put == 1:
    puzzle = cv2.imread(args["puzzle"])
    waldo = cv2.imread(args["waldo"])
    (wHeight, wWidth) = waldo.shape[:2]

    #find the waldo in the puzzle
    result = cv2.matchTemplate(puzzle, waldo, cv2.TM_CCOEFF)
    (_, _, minLoc, maxLoc) = cv2.minMaxLoc(result)

    # the puzzle image
    topLeft = maxLoc
    botRight = (topLeft[0] + wWidth, topLeft[1] + wHeight)
    roi = puzzle[topLeft[1]:botRight[1], topLeft[0]:botRight[0]]

    # construct a darkened transparent 'layer' to darken everything
    # in the puzzle except for waldo
    mask = np.zeros(puzzle.shape, dtype = "uint8")
    puzzle = cv2.addWeighted(puzzle, 0.25, mask, 0.75, 0)

    # put the original waldo back in the image so that he is
    # 'brighter' than the rest of the image
    puzzle[topLeft[1]:botRight[1], topLeft[0]:botRight[0]] = roi
    # display the images
    cv2.imshow("Puzzle", imutils.resize(puzzle, height = 650))
    cv2.waitKey(0)


