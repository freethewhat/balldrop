import cv2
import numpy as np
from random import randrange

class dot():
    def __init__(self, width, height, block):
        self.x = randrange(width/block)
        self.y = 0
        self.width = w
        self.height = h
        self.block = block

    def drawBall(self, img):
        img = cv2.circle(img, (self.x * block + block / 2, self.y * block + block / 2), self.block / 2, (0, 0, 255), thickness=-1)
        if self.y >= self.height/self.block:
            self.newBall()
        else:
            self.y += 1

        return img

    def newBall(self):
        self.x = randrange(self.width/self.block)
        self.y = 0

    def detectCollision(self, img):
        if self.y <= 0 | self.y >= self.height / self.block:
            maskValue = np.zeros((self.block,self.block))
        else:
            maskValue = img[
                self.y * self.block : self.y * self.block + self.block,
                self.x * self.block : self.x * self.block + self.block,
            ]

        maskValue = cv2.mean(maskValue)

        if maskValue[0] > 100:
            self.newBall()

cap = cv2.VideoCapture(0)
_, frame = cap.read()
h, w, _ = frame.shape

mask = np.zeros((h, w), dtype=np.uint8)
mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

block = 20
radiusOffset = 5

fgbg = cv2.createBackgroundSubtractorMOG2()

ball = dot(w, h, block)

while True:
    _, image = cap.read()
    image = cv2.flip(image, 1)
    for x in range(w/block):
        for y in range(h/block):
            temp = image[y * block : y * block + block, x * block : x * block + block]
            mean = cv2.mean(temp)
            image[y * block : y * block + block, x * block : x * block + block] = [mean[0], mean[1], mean[2]]
            cv2.rectangle(image, (x * block, y * block), (x * block + block, y * block + block), (0, 0, 0), thickness=-1)
            cv2.rectangle(mask, (x * block, y * block), (x * block + block, y * block + block), (mean[0], mean[1], mean[2]), thickness=-1)
            cv2.circle(image, (x * block + block / 2, y * block + block / 2), block / 2, (mean[0], mean[1], mean[2]), thickness=-1)

    mask = fgbg.apply(mask, 1)

    image = ball.drawBall(image)
    image = cv2.GaussianBlur(image, (5, 5), 1)

    collision = ball.detectCollision(mask)

    cv2.imshow('image', image)
    # cv2.imshow('mask', mask)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
