import numpy as np
import cv2 as cv

def dist(a, b):
    return ((b[0]-a[0])**2 + (b[1]-a[1])**2)**(1/2)

def inCircle(circles,point):
    for i in circles[0]:
        if dist([i[0],i[1]],point) < i[2]+15:
            return True
    return False

def circles(a, b):
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
    return  cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 110,
                                   param1=a, param2=b,
                                   minRadius=0, maxRadius=rows//8)

def checker(image):
    img = cv.imread(image)
    width = len(img)
    heigth = len(img[0])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray2 = np.float32(gray)
    dst = cv.cornerHarris(gray2, 2, 3, 0.04)

    dst = cv.dilate(dst, None)

    circlesss = circles(30, 40)

    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    #cv.imshow('KOX',img)
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

    if circlesss is not None:
        circlesss = np.uint16(np.around(circlesss))
        for i in circlesss[0, :]:
            center = (i[0], i[1])
            cv.circle(img, center, 1, (0, 100, 100), 3)
            radius = i[2]
            cv.circle(img, center, radius, (255, 0, 255), 3)

    szerokosc = rightdown[0]-lefttop[0]
    wysokosc = rightdown[1] - lefttop[1]

    plansza=[[0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]]
    county = 0
    countx = 0
    pionki = []
    for i in range(lefttop[1], rightdown[1], wysokosc//8):
        if(county == 8):
            break
        for j in range(lefttop[0], rightdown[0], szerokosc//8):
            if(countx == 8):
                break
            if(inCircle(circlesss,[j+szerokosc//16,i+wysokosc//16])):
                pionki.append([j,i])
                #plansza[county][countx] = gray[j+szerokosc//16,i+wysokosc//16]
                plansza[county][countx] = 1
                cv.circle(img, (j + szerokosc // 16, i + wysokosc // 16), 3, (0, 100, 100), 3)
            countx += 1
        countx=0
        county+=1
    print(len(pionki))
    for i in plansza:
        print(i)
    cv.imwrite('dst.jpg', img)

checker('boarddd.jpg')
