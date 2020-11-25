import numpy as np
import cv2 as cv

def dist(a, b):
    return ((b[0]-a[0])**2 + (b[1]-a[1])**2)**(1/2)

def inCircle(circles,point):
    for i in circles[0]:
        if dist([i[0],i[1]],point) < i[2]:
            return True
    return False

def corners2():
    img = cv.imread('checkersss.png')
    width = len(img)
    heigth = len(img[0])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)

    dst = cv.dilate(dst, None)

    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    lefttop = [width, heigth]
    righttop = [0, heigth]
    leftdown = [width, 0]
    rightdown = [0,0]

    for i in range(len(img)):
        for j in range(len(img[i])):
            if(img[i][j][2]==255) and dist([0,0], lefttop) > dist([0,0], [i, j]):
                lefttop = [i, j]
            if(img[i][j][2]==255) and dist([width,0], righttop) > dist([width,0], [i, j]):
                righttop = [i, j]
            if(img[i][j][2]==255) and dist([0,heigth], leftdown) > dist([0,heigth], [i, j]):
               leftdown = [i, j]
            if(img[i][j][2]==255) and dist([width,heigth], rightdown) > dist([width,heigth], [i, j]):
               rightdown = [i, j]
    print(lefttop, righttop, leftdown, rightdown)
    cv.imwrite('dst.jpg', img)

def corners():
    img = cv.imread('checkersss.png')

    cv.imshow('img', img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)

    dst = cv.dilate(dst, None)

    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv.imshow('dst', img)

    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

def circles(a, b):
    scale = 1
    delta = 0
    ddepth = cv.CV_16S

    img = cv.imread('unknown.png')

    #cv.imshow('img', img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #gray = cv.GaussianBlur(gray, (5,5),cv.BORDER_DEFAULT)
    #gray = cv.equalizeHist(gray)
    rows = gray.shape[0]
    kernel = np.ones((8,8),np.uint8)
    gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
    gray = cv.GaussianBlur(gray, (5,5),cv.BORDER_DEFAULT)
    gray = cv.equalizeHist(gray)
    #cv.imshow('lol', gray)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 110,
                                   param1=a, param2=b,
                                   minRadius=0, maxRadius=rows//8)

    #print(inCircle(circles,[415,415]))

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            cv.circle(img, center, 1, (0, 100, 100), 3)
            radius = i[2]
            cv.circle(img, center, radius, (255, 0, 255), 3)

    cv.imshow('lol', img)
    #cv.imwrite('result_{}_{}.jpg'.format(a, b), img)

def forcheck():
    img = cv.imread('unknown.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #gray = cv.medianBlur(gray, 5)
    #gray = cv.equalizeHist(gray)
    rows = gray.shape[0]
    kernel = np.ones((9,9),np.uint8)
    gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
    gray = cv.equalizeHist(gray)
    cv.imshow('lol', gray)


circles(50, 100)
#corners()
