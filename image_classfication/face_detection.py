import cv2 as cv2
import numpy as np

#이미지는 2비트부터 16비트까지 다양하게 존재한다. 2비트는 흑백 색깔밖에 없다.
#2비트로 이미지를 다루려면 데이터가 너무 부족하기에 2^8 인 0~255까지 다루고 그 데이터가 3개인 경우 RGB로 색깔을 다채롭게 표현할 수 있다.

img = cv2.imread('./images.jpg')

kernel = np.ones((5,5), np.uint8) #np.uint8 unsigned 8 bit == 256개의 정보를 갖고 있다는 뜻, 이미지도 0~255이기에 맞다.
#색깔 변환
imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(5,5),0 )
imgCanny = cv2.Canny(img, 150,200)
imgdilation = cv2.dilate(imgCanny, kernel, iterations=1 )#iteration == thick에 영향을 줌
imgErode = cv2.erode(imgdilation,kernel, iterations=1)

width, height = 250,350
pts1 = np.float32([[111,219],[287,188],[154,482],[352,440]])
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])

matrix = cv2.getPerspectiveTransform(pts1,pts2)
imgoutput = cv2.warpPerspective(img, matrix, (width,height))# 포인트 4개 찍은 pts1뽑아서 pts2로 만들기

def fun():
    pass
cv2.namedWindow("Trackbar")
cv2.resizeWindow("Trackbar", 640,240)
cv2.createTrackbar("Hue min", "Trackbar",0,255,fun)


while True:

    h_min = cv2.getTrackbarPos("Hue min", "Trackbar")
    print(h_min)
    #cv2.imshow("img",img)


cv2.imshow("output",imgGray)
cv2.imshow("blur", imgBlur)
cv2.imshow("canny", imgCanny)
cv2.imshow("dilation",imgdilation)
cv2.imshow("erode", imgErode)
cv2.imshow("pts1-> pts2", imgoutput)
cv2.waitKey()


'''
#이미지 보여주기, 매초 프레임당 새로운 이미지를 보여줘 동영상처럼 보인다.
cap = cv2.VideoCapture('주소')
while True:#프레임 단위당 보여준다.
    success, img = cap.read()
    cv2.imshow("video",img )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap = cv2.VideoCapture(0) #0은 내장 카메라 이용
cap.set(3,640)
cap.set(4,480)

while True:
    success, img = cap.read()
    cv2.imshow("video", img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break


'''
