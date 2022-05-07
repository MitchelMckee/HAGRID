import os
import cv2 as cv
from natsort import natsorted

capture = cv.VideoCapture(0)
path = './screenshots/'

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
    cv.imwrite(path + filename,capture.read()[1])

def runCam():
    while True:
        isTrue, frame = capture.read()
        cv.imshow('Video', frame)
        if cv.waitKey(20) & 0xFF==ord('d'):
            break
        if cv.waitKey(20) & 0xFF==ord('s'):
            screenshot()

    capture.release()
    cv.destroyAllWindows()

runCam()