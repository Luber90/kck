import numpy as np
import cv2 as cv
import math

def imageMultirescale(img): # zmniejsza o 3/4 zdjęcie
    img = cv.imread(img)
    img = cv.resize(img, ((int(img.shape[1] * 1 / 2)), int(img.shape[0] * 1 / 2)))
    cv.imwrite('zdj/inZdjjj{}.jpg'.format(i), img)

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

#def minus(b, a):
    #return [b[0]-a[0], b[1]-a[1]]

#def addd(a, b):
    #return [a[0]+b[0], a[1]+b[1]]

#def mul(a, b):
    #return [a[0]*b, a[1]*b]

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
    #cv.imshow('XD',img)
    cv.waitKey(0)
    print("angle:", angle*(180/np.pi))
    print("lefttop:")
    img2 = cv.imread(plik)
    heigth2 = len(img2)
    width2 = len(img2[0])
    width = len(img)
    heigth = len(img[0])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    _, gray = cv.threshold(gray, 20, 255, cv.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)

    #cv.imshow('XD', gray)
    cv.waitKey(0)

    kernel = np.ones((2, 2), np.uint8)
    gray = cv.erode(gray, kernel, iterations=4)

    # cv.imshow('XD', gray)
    # cv.imshow('XDDD', img2)
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

def circles2(image): # funkcja do znajdywania najodpowiedniejszego wykrywania kółek
    img = cv.imread(image)
    # img = cv.resize(img, (500, 500))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rows = gray.shape[0]
    kernel = np.ones((8, 8), np.uint8)
    gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
    gray = cv.medianBlur(gray, 5)
    gray = cv.equalizeHist(gray)



    circlesArr = []
    for i in range(80,130,5):
        for j in range(5, 65, 5):
            circlesArr.append(cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 110,
                                      param1=i, param2=j,
                                      minRadius=0, maxRadius=rows // 8))
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

            if np.std(varianceList) > rows/128: #filtracja -  zbyt różnorodne wielkości pionków
                continue

            finalCircles.append(circlesArr[c][0])
            #print(circlesArr[c][0])

            print('Odchylenie            : ', np.std(varianceList))
            #imgCpy = cv.resize(imgCpy, (500, 500))
            finalPictures.append([imgCpy, len(circlesArr[c][0]), int(np.var(varianceList))]) #obrazek i ilość kółek

    finalCirclesRet = []
    finalPicturesRet = []
    maxCircles = 0
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
    thetas = []
    blank_image = np.zeros((500, 500, 3), np.uint8)
    if strong_lines is not None:
        for i in range(0, len(strong_lines)):
            rho = strong_lines[i][0][0]
            theta = strong_lines[i][0][1]
            if theta*(180/np.pi) > -45 and theta*(180/np.pi) < 60:
                #print(theta*(180/np.pi))
                thetas.append(theta)
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            avg, var, rozniceavg, dista = linepoints(src, strong_lines[i], gray)
            #print(dista)
            #if math.sqrt(var) > 3 and dista < 275:
            cv.line(blank_image, pt1, pt2, (0, 0, 255), 2)
    print(thetas)
    return blank_image, np.average(thetas)

def linepoints(img, line, gray):
    rho = line[0][0]
    theta = line[0][1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    zbior = []
    roznice = []
    dista = []
    for i in range(0, 520, 30):
        pt = (int(x0+i*(-b)), int(y0+i*a))
        if pt[0] < 500 and pt[0] > -1 and pt[1] < 500 and pt[1] > -1:
            zbior.append(gray[pt[0]][pt[1]])
            dista.append(dist(pt, [249, 249]))
            if pt[0] + 5 < 500 and pt[0] - 5 > -1:
                roznice.append(abs(int(gray[pt[0]+5][pt[1]]) - int(gray[pt[0]-5][pt[1]])))
            if pt[1] + 5 < 500 and pt[1] - 5 > -1:
                roznice.append(abs(int(gray[pt[0]][pt[1]+5]) - int(gray[pt[0]][pt[1]-5])))
            #cv.circle(img, pt, 2, (255, 0, 255), 3)
    for i in range(0, -520, -30):
        pt = (int(x0 + i * (-b)), int(y0 + i * a))
        if pt[0] < 500 and pt[0] > -1 and pt[1] < 500 and pt[1] > -1:
            zbior.append(gray[pt[0]][pt[1]])
            dista.append(dist(pt, [249, 249]))
            if pt[0] + 5 < 500 and pt[0] - 5 > -1:
                roznice.append(abs(int(gray[pt[0] + 5][pt[1]]) - int(gray[pt[0] - 5][pt[1]])))
            if pt[1] + 5 < 500 and pt[1] - 5 > -1:
                roznice.append(abs(int(gray[pt[0]][pt[1] + 5]) - int(gray[pt[0]][pt[1] - 5])))
            #cv.circle(img, pt, 1, (255, 0, 255), 1)
    return np.average(zbior), np.var(zbior), np.average(roznice), np.average(dista)

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
                    interpole(i/8+1/8, j/8, lefttop, righttop, leftdown, rightdown), (255, 0, 0), 1)
            cv.line(img, interpole(i/8, j/8, lefttop, righttop, leftdown, rightdown),
                    interpole(i/8, j/8+1/8, lefttop, righttop, leftdown, rightdown), (255, 0, 0), 1)
            if inCircle(circles, interpole(i/8+1/16, j/8+1/16, lefttop, righttop, leftdown, rightdown)):
                pionki.append([j, i])
                plansza[j][i] = 1
                cv.circle(img, interpole(i/8+1/16, j/8+1/16, lefttop, righttop, leftdown, rightdown), 3, (0, 255, 0), 3)
    cv.line(img, tuple(leftdown), tuple(rightdown), (255, 0, 0), 1)
    cv.line(img, tuple(rightdown), tuple(righttop),(255, 0, 0), 1)
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
    cv.imwrite('final.jpg', img)
    cv.imwrite('final{}.jpg'.format(cunt), img)
    cv.waitKey()


cunt = 0

#plik = 'zdj/inZdjjj{}.jpg'.format(i)

for DDD in range(12,42):
    cunt = DDD
    plik = 'zdj/inZdjjj{}.jpg'.format(DDD)
    print('ZDJ numero : ', DDD, '  :', plik, "<<<<<==================================================")
    zdj, angle = lines(plik, 50, 40)
    lefttop, righttop, leftdown, rightdown = corners2(zdj, plik, angle)
    #zoba(plik, lefttop, righttop, leftdown, rightdown)
    circless = circles2(plik)
    final(plik, circless, lefttop, righttop, leftdown, rightdown)

    cunt+=1


# for i in range(12,37):
#     img = 'zdj/inZdjjj{}.jpg'.format(35)
#     imageMultirescale(img)