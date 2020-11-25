import numpy as np
import cv2 as cv



def dist(a, b):
    return ((b[0]-a[0])**2 + (b[1]-a[1])**2)**(1/2)

img = cv.imread('unknown.png')
width = len(img)
heigth = len(img[0])
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray2 = np.float32(gray)
dst = cv.cornerHarris(gray2, 2, 3, 0.04)

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
        if(gray[j+szerokosc//16][i+wysokosc//16]<100):
            pionki.append([j,i])
            plansza[countx][county] = 1
        countx += 1
    countx=0
    county+=1
print(len(pionki))
for i in plansza:
    print(i)
cv.imwrite('dst.jpg', gray)


