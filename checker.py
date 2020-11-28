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

def contures():
    # Let's load a simple image with 3 black squares
    image = cv.imread('boarddd.jpg')
    cv.waitKey(0)

    # Grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Find Canny edges
    edged = cv.Canny(gray, 30, 200)
    cv.waitKey(0)

    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the image
    contours, hierarchy = cv.findContours(edged,
                                           cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    cv.imshow('Canny Edges After Contouring', edged)
    cv.waitKey(0)

    print("Number of Contours found = " + str(len(contours)))

    # Draw all contours
    # -1 signifies drawing all contours
    cv.drawContours(image, contours, -1, (0, 255, 0), 3)

    cv.imwrite('dsttt.jpg', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
def sudoku():
    import cv2
    import numpy as np

    image = cv2.imread("checker1.jpg")
    cv2.imshow("Image", image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow("blur", blur)

    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    cv2.imshow("thresh", thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    c = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000:
            if area > max_area:
                max_area = area
                best_cnt = i
                image = cv2.drawContours(image, contours, c, (0, 255, 0), 3)
        c += 1

    mask = np.zeros((gray.shape), np.uint8)
    cv2.drawContours(mask, [best_cnt], 0, 255, -1)
    cv2.drawContours(mask, [best_cnt], 0, 0, 2)
    cv2.imshow("mask", mask)

    out = np.zeros_like(gray)
    out[mask == 255] = gray[mask == 255]
    cv2.imshow("New image", out)

    blur = cv2.GaussianBlur(out, (5, 5), 0)
    cv2.imshow("blur1", blur)

    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    cv2.imshow("thresh1", thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    c = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000 / 2:
            cv2.drawContours(image, contours, c, (0, 255, 0), 3)
        c += 1

    cv2.imwrite("Final Image.jpg", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#checker('boarddd.jpg')
#contures()
sudoku()