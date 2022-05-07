import os
import cv2
import numpy as np
from natsort import natsorted

capture = cv2.VideoCapture(0)
path = './screenshots/'
global filename

def newFileName():
    folder = os.fsencode(path)
    filenames = []
    global filename

    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith(('.png')):
            filenames.append(filename)

    filenames = (natsorted(filenames))
    filename = filenames[-1]
    filename.split('.')
    x = int(filename[0]) + 1
    filename = str(x) + '.png'

def screenshot():
    print('SCREENSHOT')
    newFileName()
    cv2.imwrite(path + filename,capture.read()[1])

def runCam():

    
    while True:
        (ret, frame) = capture.read()
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteFrame) = cv2.threshold(grayFrame, 127, 255, cv2.THRESH_BINARY)
    
        isTrue, frame = capture.read()
        cv2.imshow('Video', blackAndWhiteFrame)
        if cv2.waitKey(20) & 0xFF==ord('d'):
            break
        if cv2.waitKey(20) & 0xFF==ord('s'):
            screenshot()

    capture.release()
    cv2.destroyAllWindows()

def boundLetter(img):
    image = cv2.imread('./screenshots/' + img)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ROI_number = 0
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite('1.png')

    cv2.imshow('Image', image)
    cv2.waitKey()
    
def takeSSandCreateBoundingBoxes():
    runCam()
    print(filename)
    print("Bounding")
    boundLetter(filename)
#runCam()
#boundLetter()
takeSSandCreateBoundingBoxes()