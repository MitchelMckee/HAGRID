import os
import cv2
import numpy as np
from natsort import natsorted # USED TO SORT THE FILES IN NATURAL ORDER 

capture = cv2.VideoCapture(0)
path = './screenshots/'
global filename

def newFileName(): # GET THE NEXT NUMBER AVAILABLE IN THE FILE, THE NAME OF THE FILE WILL ITERATE
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

def screenshot(): # TAKE A SCREENSHOT FROM THE CAMERA
    print('SCREENSHOT')
    newFileName()
    cv2.imwrite(path + filename,capture.read()[1])

def runCam(): # RUN THE CAMERA TO SEE WHAT YOU ARE TRYING TO SCAN
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

def boundLetter(img): # CREATE BOUNDING BOXES
   #image = cv2.imread('./predict/*.png') # CHANGE ASTERISK TO PREDICTION FILE IF YOU ALREADY HAVE A PICTURE
    image = cv2.imread('./screenshots/' + img) # USE CAMERA AS IMAGE TO PREDICT
    original = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ROI_number = 0
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts: # CREATE BOUNDING BOXES AND SAVE EACH BOX AS AN IMAGE FILE
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2) # DRAW A RECTANGLE AROUND EACH SHAPE FOUND
        ROI = original[y:y+h, x:x+w]
        cv2.imwrite('./predict/Image_{}.png'.format(ROI_number), ROI) # SAVES EACH BOX AS AN IMAGE WITH AN ITERATING NAME
        ROI_number += 1

    cv2.imshow('Image', image)
    cv2.imwrite('./bounded/' + filename, image)
    cv2.waitKey()
    
def takeSSandCreateBoundingBoxes(): # TAKE A SCREENSHOT FROM THE CAMERA, SAVE IT, GIVE IT BOUNDING BOXES AND SAVE THOSE BOXES
    runCam()
    print(filename)
    print("Bounding")
    boundLetter(filename)

takeSSandCreateBoundingBoxes()