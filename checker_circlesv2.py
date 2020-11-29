import numpy as np
import cv2 as cv
import math


def dist(a, b):
    return ((b[0]-a[0])**2 + (b[1]-a[1])**2)**(1/2)

def inCircle(circles,point):
    for i in circles[0]:
        if dist([i[0],i[1]],point) < i[2]:
            return True
    return False

def wrongCircles(circles,tr,bl): #odrzuca zdjęcia z wykrytymi kółkami różnych rozmiarów
    dst = dist(tr,bl)
    for i in circles[0]:
        if i[2] > dst/16:
            return True
    return False

def corners2():
    img = cv.imread('boaard.png')
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

    return righttop, leftdown

def corners(image):
    img = cv.imread(image)

    cv.imshow('img', img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 3, 3, 0.04)

    dst = cv.dilate(dst, None)

    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv.imshow('dst', img)
    cv.imwrite('tegotamtego.jpg', img)

    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

def circles(a, b,tr, bl):
    scale = 1
    delta = 0
    ddepth = cv.CV_16S

    img = cv.imread('boaard.png')

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

    if (circles is None):
        print("za mamło biongów")
        return "za mamło biongów"
    if (len(circles) > 24 ):
        print("Dłumgość:", len(circles), "za dumżo biongów")
        return "za dumżo biongów"
    if (wrongCircles(circles,tr, bl) is True):
        print("za dumże gółmka")
        return "za dumże gółmka"


    #print(inCircle(circles,[415,415]))

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            cv.circle(img, center, 1, (0, 100, 100), 3)
            radius = i[2]
            cv.circle(img, center, radius, (255, 0, 255), 3)
    #cv.imshow('lol', img)
    cv.imwrite('result_{}_{}.jpg'.format(a, b), img)

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

def lines(name, a, b):
    src = cv.imread(name)
    src = cv.resize(src, (500, 500))
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    dst = cv.Canny(src, a, b)
    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    strong_lines = []
    avg = 15
    for i in lines:  # recalculating parameters for lines that have negative rho
        if i[0][0] < 0:
            i[0][0] *= -1.0
            i[0][1] -= np.pi

    for i in lines:  # finding four strong (that means they have the most points) and different lines
        chk = 0
        if len(strong_lines) == 0:
            strong_lines.append(i)
            continue
        for j in strong_lines:
            if (j[0][0] - avg <= i[0][0] <= j[0][0] + avg) and (
                    j[0][1] - np.pi / 18 <= i[0][1] <= j[0][1] + np.pi / 18):
                chk += 1
        if chk == 0:
            strong_lines.append(i)


    if strong_lines is not None:
        for i in range(0, len(strong_lines)):
            rho = strong_lines[i][0][0]
            theta = strong_lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            avg, var = linepoints(src, strong_lines[i], gray)
            print(avg, var)
            if avg > 90 and var > 300:
                cv.line(src, pt1, pt2, (0, 0, 255), 1, cv.LINE_AA)
    cv.imwrite('lol2.jpg', src)
    cv.waitKey()


def linepoints(img, line, gray):
    rho = line[0][0]
    theta = line[0][1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    zbior = []
    for i in range(0, 520, 20):
        pt = (int(x0+i*(-b)), int(y0+i*a))
        if pt[0] < 500 and pt[0] > -1 and pt[1] < 500 and pt[1] > -1:
            zbior.append(gray[pt[0]][pt[1]])
            #cv.circle(img, pt, 2, (255, 0, 255), 3)
    for i in range(0, -520, -20):
        pt = (int(x0 + i * (-b)), int(y0 + i * a))
        if pt[0] < 500 and pt[0] > -1 and pt[1] < 500 and pt[1] > -1:
            zbior.append(gray[pt[0]][pt[1]])
            #cv.circle(img, pt, 2, (255, 0, 255), 3)
    return np.average(zbior), np.var(zbior)

# count = 0
#
# tr, bl = corners2()
# for i in range(30,200,5):
#     for j in range(30, 200, 5):
#         print(j,i)
#         circles(j,i,tr, bl)
#         count+=1
# print (count)
#circles(30,55)

lines('boarddd.jpg', 50, 40)
