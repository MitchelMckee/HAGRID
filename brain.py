#  @InProceedings{deCampos09,
#  author    = "de Campos, T.~E. and Babu, B.~R. and Varma, M.",
#  title     = "Character recognition in natural images",
#  booktitle = "Proceedings of the International Conference on Computer
#  Vision Theory and Applications, Lisbon, Portugal",
#  year      = "2009",
#  month     = "February",
#}

#from __future__ import print_function
import numpy as np
import keras
import cv2
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras import backend as K

#SAVE MODEL - TODO
import pandas
import joblib
import pickle

#GRAPHING
import matplotlib.pyplot as plt

#PARAMS
letter_classes = 26
batch_size = 128
epochs = 5

#DATASET PARAMS
IMG_FOLDER = r'/Users/mitchelmckee/Desktop/HAGRID/dataset/'
IMG_SIZE = 50
CATEGORIES = ['a', 'b', 'c']

for category in CATEGORIES:
    path = os.path.join(IMG_FOLDER, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        plt.imshow(img_array)
        break
    break

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array)

training_data = []
def create_training_data():
    for i in range(len(CATEGORIES)):
        category = CATEGORIES[i]
        path = os.path.join(IMG_FOLDER, category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img))
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, i])
            
create_training_data()



