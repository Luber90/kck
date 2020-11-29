import numpy as np
import cv2 as cv
import math


def dist(a, b):
    return ((b[0]-a[0])**2 + (b[1]-a[1])**2)**(1/2)

def inCircle(circles,point):
    for i in circles[0]:
        if dist([i[0],i[1]],point) < i[2] + 23:
            return True
    return False

def wrongCircles(circles,tr,bl): #odrzuca zdjęcia z wykrytymi kółkami różnych rozmiarów
    dst = dist(tr,bl)
    for i in circles[0]:
        if i[2] > dst/16:
            return True
    return False

def corners2(img, plik):
    img2 = cv.imread(plik)
    width2 = len(img2)
    heigth2 = len(img2[0])
    width = len(img)
    heigth = len(img[0])
    #img = cv.resize(img, (width, heigth))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)

    #dst = cv.dilate(dst, None)

    img[dst > 0.01 * dst.max()] = [0, 255, 0]

    lefttop = [width, heigth]
    righttop = [0, heigth]
    leftdown = [width, 0]
    rightdown = [0,0]

    for i in range(1, len(img)-1):
        for j in range(1, len(img[i])-1):
            if(img[i][j][1]==255) and dist([0,0], lefttop) > dist([0,0], [i, j]):
                lefttop = [i, j]
            if(img[i][j][1]==255) and dist([width,0], righttop) > dist([width,0], [i, j]):
                righttop = [i, j]
            if(img[i][j][1]==255) and dist([0,heigth], leftdown) > dist([0,heigth], [i, j]):
               leftdown = [i, j]
            if(img[i][j][1]==255) and dist([width,heigth], rightdown) > dist([width,heigth], [i, j]):
               rightdown = [i, j]
    lefttop[0] = int(lefttop[0]/width*width2)
    lefttop[1] = int(lefttop[1] / heigth* heigth2)
    righttop[0] = int(righttop[0] / width * width2)
    righttop[1] = int(righttop[1] / heigth * heigth2)
    leftdown[0] = int(leftdown[0] / width * width2)
    leftdown[1] = int(leftdown[1] / heigth * heigth2)
    rightdown[0] = int(rightdown[0] / width * width2)
    rightdown[1] = int(rightdown[1] / heigth * heigth2)
    print(lefttop, righttop, leftdown, rightdown)
    cv.imwrite('dst.jpg', img)

    return lefttop, righttop, leftdown, rightdown

def circles(name, a, b):
    img = cv.imread(name)
    #img = cv.resize(img, (500, 500))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rows = gray.shape[0]
    kernel = np.ones((8,8),np.uint8)
    gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
    gray = cv.medianBlur(gray, 5)
    gray = cv.equalizeHist(gray)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 110,
                                   param1=a, param2=b,
                                   minRadius=0, maxRadius=rows//8)


    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            cv.circle(img, center, 1, (0, 100, 100), 3)
            radius = i[2]
            cv.circle(img, center, radius, (255, 0, 255), 3)
    #cv.imshow('lol', img)
    #cv.waitKey()
    img = cv.resize(img, (500, 500))
    cv.imwrite('result_{}_{}.jpg'.format(a, b), img)
    return circles

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

    blank_image = np.zeros((500, 500, 3), np.uint8)
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
            avg, var, rozniceavg = linepoints(src, strong_lines[i], gray)
            print(avg, var, rozniceavg)
            if rozniceavg > 10:
                cv.line(blank_image, pt1, pt2, (0, 0, 255), 1)
    #linepoints(src, strong_lines[0], gray)
    #cv.imwrite("trash.jpg", thresh1)
    #cv.imshow('l', blank_image)
    #cv.imwrite('lol2.jpg', src)
    return blank_image
    cv.waitKey()

def linepoints(img, line, gray):
    rho = line[0][0]
    theta = line[0][1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    zbior = []
    roznice = []
    for i in range(0, 520, 30):
        pt = (int(x0+i*(-b)), int(y0+i*a))
        if pt[0] < 500 and pt[0] > -1 and pt[1] < 500 and pt[1] > -1:
            zbior.append(gray[pt[0]][pt[1]])
            if pt[0] + 5 < 500 and pt[0] - 5 > -1:
                roznice.append(abs(int(gray[pt[0]+5][pt[1]]) - int(gray[pt[0]-5][pt[1]])))
            if pt[1] + 5 < 500 and pt[1] - 5 > -1:
                roznice.append(abs(int(gray[pt[0]][pt[1]+5]) - int(gray[pt[0]][pt[1]-5])))
            #cv.circle(img, pt, 2, (255, 0, 255), 3)
    for i in range(0, -520, -30):
        pt = (int(x0 + i * (-b)), int(y0 + i * a))
        if pt[0] < 500 and pt[0] > -1 and pt[1] < 500 and pt[1] > -1:
            zbior.append(gray[pt[0]][pt[1]])
            if pt[0] + 5 < 500 and pt[0] - 5 > -1:
                roznice.append(abs(int(gray[pt[0] + 5][pt[1]]) - int(gray[pt[0] - 5][pt[1]])))
            if pt[1] + 5 < 500 and pt[1] - 5 > -1:
                roznice.append(abs(int(gray[pt[0]][pt[1] + 5]) - int(gray[pt[0]][pt[1] - 5])))
            #cv.circle(img, pt, 1, (255, 0, 255), 1)
    return np.average(zbior), np.var(zbior), np.average(roznice)

def zoba(plik, a, b, c, d):
    src = cv.imread(plik)
    src = cv.resize(src, (500, 500))
    cv.circle(src, tuple(a), 2, (255, 0, 255), 3)
    cv.circle(src, tuple(b), 2, (255, 0, 255), 3)
    cv.circle(src, tuple(c), 2, (255, 0, 255), 3)
    cv.circle(src, tuple(d), 2, (255, 0, 255), 3)
    cv.imshow("zoa", src)
    cv.waitKey()

def final(name, circles, lefttop, righttop, leftdown, rightdown):
    img = cv.imread(name)
    #img = cv.resize(img, (500, 500))
    szerokosc = rightdown[0] - lefttop[0]
    wysokosc = rightdown[1] - lefttop[1]

    plansza = [[0, 0, 0, 0, 0, 0, 0, 0],
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
    for i in range(lefttop[1], rightdown[1], wysokosc // 8):
        if (county == 8):
            break
        for j in range(lefttop[0], rightdown[0], szerokosc // 8):
            if (countx == 8):
                break
            if (inCircle(circles, [j + szerokosc // 16, i + wysokosc // 16])):
                pionki.append([j, i])
                # plansza[county][countx] = gray[j+szerokosc//16,i+wysokosc//16]
                plansza[county][countx] = 1
                cv.circle(img, (j + szerokosc // 16, i + wysokosc // 16), 3, (0, 100, 100), 3)
            countx += 1
        countx = 0
        county += 1
    print(len(pionki))
    for i in plansza:
        print(i)
    cv.imwrite('final.jpg', img)
    cv.waitKey()

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

plik = 'boaard.png'

zdj = lines(plik, 50, 40)
lefttop, righttop, leftdown, rightdown = corners2(zdj, plik)
#zoba(plik, lefttop, righttop, leftdown, rightdown)
circless = circles(plik, 90, 30)
final(plik, circless, lefttop, righttop, leftdown, rightdown)


