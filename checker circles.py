import numpy as np
import cv2 as cv
scale = 1
delta = 0
ddepth = cv.CV_16S

img = cv.imread('checker1.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.medianBlur(gray, 5)
gray = cv.equalizeHist(gray)
rows = gray.shape[0]
circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 120,
                               param1=275, param2=98,
                               minRadius=10, maxRadius=0)
    
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        cv.circle(img, center, 1, (0, 100, 100), 3)
        radius = i[2]
        cv.circle(img, center, radius, (255, 0, 255), 3)

cv.imshow('lol', img)


