import numpy as np
import cv2 as cv
import math

def imageMultirescale(img): # zmniejsza o 3/4 zdjęcie
    img = cv.imread(img)
    img = cv.resize(img, ((int(img.shape[1] * 1 / 2)), int(img.shape[0] * 1 / 2)))
    cv.imwrite('zdj/inZdjjj.jpg', img)

def interpole(v, u, p1, p2, p3, p4): #np ile w prawo [0,1], nd ile w dol[0,1], v1 gorny wektro w prawo, v2 dolny wektor w prawo, v3 lewy wektor w dol itp
    return (int((1-v)*((1-u)*p1[0]+u*p3[0])+v*((1-u)*p2[0]+u*p4[0])),
            int((1-v)*((1-u)*p1[1]+u*p3[1])+v*((1-u)*p2[1]+u*p4[1])))

def minn(a, b):
    if a <= b:
        return a
    else:
        return b

def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return int(qx), int(qy)

def dist(a, b):
    return ((b[0]-a[0])**2 + (b[1]-a[1])**2)**(1/2)

def inCircle(circles,point):
    for i in circles[0]:
        if dist([i[0],i[1]],point) < i[2]+10:
            return True
    return False

def wrongCircles(circles,tr,bl): #odrzuca zdjęcia z wykrytymi kółkami różnych rozmiarów
    dst = dist(tr,bl)
    for i in circles[0]:
        if i[2] > dst/16:
            return True
    return False

def corners2(img, plik, angle):
    print("angle:", angle*(180/np.pi))
    print("lefttop:")
    img2 = cv.imread(plik)
    heigth2 = len(img2)
    width2 = len(img2[0])
    width = len(img)
    heigth = len(img[0])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img1 = np.copy(gray)
    gray = cv.medianBlur(gray, 5)
    _, gray = cv.threshold(gray, 20, 255, cv.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel, iterations=1)
    img2 = np.copy(gray)
    #cv.imshow('XD', gray)
    cv.waitKey(0)

    kernel = np.ones((2, 2), np.uint8)
    gray = cv.erode(gray, kernel, iterations=4)
    img3 = np.copy(gray)
    # cv.imshow('XD', gray)
    # cv.imshow('XDDD', img2)
    print(img1.shape, img2.shape)
    res = cv.hconcat([img1, img2])
    cv.imwrite("nokoniec.jpg", cv.hconcat([res, img3]))
    cv.waitKey(0)

    img[gray==255] = (255,255,255)
    center = [249, 249]
    lefttop = rotate(center, [width, heigth], angle)
    righttop = rotate(center, [0, heigth], angle)
    leftdown = rotate(center, [width, 0], angle)
    rightdown = rotate(center, [0,0], angle)
    corner1 = rotate(center, [0,0], angle)
    corner2 = rotate(center, [width, 0], angle)
    corner3 = rotate(center, [0, heigth], angle)
    corner4 = rotate(center, [width, heigth], angle)
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
            #if(gray[i][j] == 255):
                #cv.circle(gray, (j, i), 4, (100, 0, 0))
            if(gray[i][j]==255) and dist(corner1, lefttop) > dist(corner1, [j, i]):
                lefttop = [j, i]
            if(gray[i][j]==255) and dist(corner2, righttop) > dist(corner2, [j, i]):
                righttop = [j, i]
            if(gray[i][j]==255) and dist(corner3, leftdown) > dist(corner3, [j, i]):
               leftdown = [j, i]
            if(gray[i][j]==255) and dist(corner4, rightdown) > dist(corner4, [j, i]):
               rightdown = [j, i]
    cv.circle(img, (lefttop[0], lefttop[1]), 4, (255, 0, 0), 2)
    cv.circle(img, (rightdown[0], rightdown[1]), 4, (255, 0, 0), 2)
    cv.circle(img, (righttop[0], righttop[1]), 4, (255, 0, 0), 2)
    cv.circle(img, (leftdown[0], leftdown[1]), 4, (255, 0, 0), 2)
    print(lefttop, righttop, leftdown, rightdown)
    lefttop[0] = int(lefttop[0]/(width-1)*(width2-1))
    lefttop[1] = int(lefttop[1] / (heigth-1)*(heigth2-1))
    righttop[0] = int(righttop[0] / (width-1)*(width2-1))
    righttop[1] = int(righttop[1] / (heigth-1)*(heigth2-1))
    leftdown[0] = int(leftdown[0] / (width-1)*(width2-1))
    leftdown[1] = int(leftdown[1] / (heigth-1)*(heigth2-1))
    rightdown[0] = int(rightdown[0] / (width-1)*(width2-1))
    rightdown[1] = int(rightdown[1] / (heigth-1)*(heigth2-1))
    print(lefttop, righttop, leftdown, rightdown)
    cv.imwrite('dst.jpg', img)

    return lefttop, righttop, leftdown, rightdown

def circles2(image): # funkcja do znajdywania najodpowiedniejszego wykrywania kółek
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rows = gray.shape[0]
    kernel = np.ones((3, 3), np.uint8)
    gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel, iterations=5)
    gray = cv.equalizeHist(gray)  # -- nie zgrywa sie z canny

    #gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
    #gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
    # gray2 = cv.morphologyEx(gray, cv.MORPH_ERODE, kernel)
    #gray = cv.Canny(gray,25,60) #-- canny sie nie nadajae bo houghCircles to debil
    #gray = cv.dilate(gray, kernel, borderType=cv.BORDER_CONSTANT) #-- useless
    #gray = cv.medianBlur(gray, 55) #-- szmata nie warto

    # cv.imshow('open',gray1)
    # gray = cv.resize(gray, (700, 700))
    #
    # cv.imshow('close{}'.format(DDD), gray)
    #
    # #cv.waitKey()
    # return False
    parameters = []
    circlesArr = []
    for i in range(50,110,5):
        for j in range(25, 60, 2):
            circlesArr.append(cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 110,
                                      param1=i, param2=j,
                                      minRadius=0, maxRadius=rows // 8))
            parameters.append([i,j])
            print(i,j)

    finalPictures = []
    finalCircles = []

    for c in range(len(circlesArr)):
        if circlesArr[c] is not None:

            if len(circlesArr[c][0]) > 24: #filtracja -  za dużo pionków
                print('Wykryte kółka za dużo!: ' , len(circlesArr[c][0]))
                continue

            print('Wykryte kółka         : ', len(circlesArr[c][0]))
            circles = np.uint16(np.around(circlesArr[c]))

            varianceList = []
            imgCpy = img.copy()
            for i in circles[0, :]:
                center = (i[0], i[1])
                cv.circle(imgCpy, center, 1, (0, 100, 100), 3)
                radius = i[2]
                varianceList.append(radius)
                cv.circle(imgCpy, center, radius, (255, 0, 255), 3)

            # if np.var(varianceList) > 100: #filtracja -  zbyt różnorodne wielkości pionków
            #     continue
            print('Odchylenie            : ', np.std(varianceList))
            if np.std(varianceList) > rows/150: #filtracja -  zbyt różnorodne wielkości pionków
                continue

            finalCircles.append(circlesArr[c][0])
            #print(circlesArr[c][0])


            #imgCpy = cv.resize(imgCpy, (500, 500))
            finalPictures.append([imgCpy, len(circlesArr[c][0]), int(np.var(varianceList)),parameters[c]]) #obrazek i ilość kółek

    finalCirclesRet = []
    finalPicturesRet = []
    maxCircles = 0
    finalParam = []
    # pozbywanie się wyników z mniejszą ilością kółek niż max
    for i in range(len(finalPictures)):
        if finalPictures[i][1] >= maxCircles:
            maxCircles = finalPictures[i][1]
    print('Znalezione kółka: ',maxCircles)

    for i in range(len(finalPictures)):
        if finalPictures[i][1] >= maxCircles:
            cv.imwrite('okDoomerFinale{}.jpg'.format(i + 1000), finalPictures[i][0])
            finalCirclesRet.append(finalCircles[i])
            finalPicturesRet.append(finalPictures[i])
            finalParam.append(finalPictures[i][3])

    print('PARAMIETRY::::::::::::::::::::::::::::::: ->   ',finalParam)

    # szukanie zbioru kółek z najmniejszą wariancją
    minVar = rows
    for i in range(len(finalPicturesRet)):
        if finalPicturesRet[i][2] <= minVar:
            minVar = finalPicturesRet[i][2]
    print("Wariancja: ",minVar)

    for i in range(len(finalPicturesRet)):
        if finalPicturesRet[i][2] == minVar:
            cv.imwrite('BestOfokDoomerFinale.jpg', finalPicturesRet[i][0])
            return [finalCirclesRet[i]]

    print('NIE ZNALEZIONO SENSOWEJ INTERPRETACJI ZDJĘCIA')
    return False
    # cv.imshow('lol', img)
    # cv.waitKey()

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
    _, gray = cv.threshold(gray, 90, 255, cv.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    gray = cv.dilate(gray, kernel, iterations=1)
    kernel = np.ones((6, 6), np.uint8)
    gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
    dst = cv.Canny(gray, a, b)
    lines = cv.HoughLines(dst, 1, np.pi / 180, 75, None, 0, 0)
    better_lines = []
    d = 15

    for i in lines:  # recalculating parameters for lines that have negative rho
        if i[0][0] < 0:
            i[0][0] *= -1.0
            i[0][1] -= np.pi

    for i in lines:
        chk = 0
        if len(better_lines) == 0:
            better_lines.append(i)
            continue
        for j in better_lines:
            if (j[0][0] - d <= i[0][0] <= j[0][0] + d) and (
                    j[0][1] - np.pi / 18 <= i[0][1] <= j[0][1] + np.pi / 18):
                chk += 1
        if chk == 0:
            better_lines.append(i)


    thetas = []
    blank_image = np.zeros((500, 500, 3), np.uint8)
    if better_lines is not None:
        for i in range(0, len(better_lines)):
            rho = better_lines[i][0][0]
            theta = better_lines[i][0][1]
            print(theta * (180 / np.pi))
            if(theta*(180/np.pi) < -2.0):
                tmp = theta
                tmp += np.pi
            else:
                tmp = theta
            if tmp*(180/np.pi) >= 0 and tmp*(180/np.pi) < 85:
                if theta < 0:
                    tmp = theta * -1.0
                    tmp -= np.pi
                    thetas.append(theta+np.pi)
                else:
                    thetas.append(theta)
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(blank_image, pt1, pt2, (0, 0, 255), 2)
    return blank_image, np.average(thetas)

def final(name, circles, lefttop, righttop, leftdown, rightdown):
    img = cv.imread(name)
    plansza = [[0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0]]

    pionki = []
    for i in range(8):
        for j in range(8):
            #print(addd(lefttop, addd(mul(prawov, j), addd(mul(dolv, i), tocenterv))))
            cv.line(img, interpole(i/8, j/8, lefttop, righttop, leftdown, rightdown),
                    interpole(i/8+1/8, j/8, lefttop, righttop, leftdown, rightdown), (255, 0, 0), 2)
            cv.line(img, interpole(i/8, j/8, lefttop, righttop, leftdown, rightdown),
                    interpole(i/8, j/8+1/8, lefttop, righttop, leftdown, rightdown), (255, 0, 0), 2)
            if inCircle(circles, interpole(i/8+1/16, j/8+1/16, lefttop, righttop, leftdown, rightdown)):
                pionki.append([j, i])
                plansza[j][i] = 1
                cv.circle(img, interpole(i/8+1/16, j/8+1/16, lefttop, righttop, leftdown, rightdown), 3, (0, 255, 0), 3)
    cv.line(img, tuple(leftdown), tuple(rightdown), (255, 0, 0), 2)
    cv.line(img, tuple(rightdown), tuple(righttop),(255, 0, 0), 2)
    print(len(pionki))
    for i in plansza:
        print(i)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            cv.circle(img, center, 1, (0, 100, 100), 3)
            radius = i[2]
            cv.circle(img, center, radius, (0, 0, 255), 3)
    cv.circle(img, tuple(righttop), 3, (0, 0, 255), 3)
    cv.circle(img, tuple(rightdown), 3, (0, 0, 255), 3)
    cv.circle(img, tuple(lefttop), 3, (0, 0, 255), 3)
    cv.circle(img, tuple(leftdown), 3, (0, 0, 255), 3)
    backupimg = cv.imread(name)
    #img = cv.resize(img, (backupimg.shape[0]))
    res = cv.hconcat([backupimg, img])
    #cv.imshow("kurwaaa", res)
    #cv.waitKey()
    cv.imwrite('C:/Users/Luber/Desktop/kck/kck/result/final_{}'.format(name[4:]), res)
    #cv.imwrite('final_{}'.format(name[4:]), res)
    # cv.imwrite('final{}.jpg'.format(str(cunt)+" close zamiast open"), img)
    # cv.imwrite('final{}.jpg'.format(str(cunt) + " morph gradient"), img)
    #cv.imwrite('STOPfinal{}.jpg'.format(str(cunt) + " morph gradient"), img)

    #cv.waitKey()

def linijka(name, a, b):
    src = cv.imread(name)
    img1 = np.copy(src)
    src = cv.resize(src, (500, 500))
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    _, gray = cv.threshold(gray, 80, 255, cv.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    gray = cv.dilate(gray, kernel, iterations=1)
    img2 = np.copy(gray)
    kernel = np.ones((8,8), np.uint8)
    gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
    img3 = np.copy(gray)
    cv.imshow('l', gray)
    cv.waitKey()
    dst = cv.Canny(gray, a, b)
    cv.imwrite("canny{}_{}.jpg".format(a, b), dst)
    lines = cv.HoughLines(dst, 1, np.pi / 180, 75, None, 0, 0)
    for i in lines:  # recalculating parameters for lines that have negative rho
        if i[0][0] < 0:
            i[0][0] *= -1.0
            i[0][1] -= np.pi
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(src, pt1, pt2, (0, 0, 255), 2)
    cv.imshow('l', src)
    cv.waitKey()

cunt = 0

plik = 'zdj/a1.jpg'

#linijka(plik, 90, 200)



#linijka(plik, 90, 200)
zdj, angle = lines(plik, 200, 90)
lefttop, righttop, leftdown, rightdown = corners2(zdj, plik, angle)

#circless = circles2(plik)
if circless is False:
    cunt += 1
final(plik, circless, lefttop, righttop, leftdown, rightdown)

'''
for DDD in [5, 6, 8, 9, 11, 23, 24, 25, 26, 27,
            33, 34, 36, 42, 43]:
    cunt = DDD
    plik = 'zdj/a{}.jpg'.format(DDD)
    print('ZDJ numero : ', DDD, '  :', plik, "<<<<<==================================================")
    zdj, angle = lines(plik, 50, 40)
    lefttop, righttop, leftdown, rightdown = corners2(zdj, plik, angle)

    circless = circles2(plik)
    if circless is False:
        cunt += 1
        continue
    final(plik, circless, lefttop, righttop, leftdown, rightdown)

    cunt+=1

cv.waitKey()
'''


