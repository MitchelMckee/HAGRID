from ast import excepthandler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.utils import np_utils

import keras
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt


# PARAMS AND GLOBALS
IMG_FOLDER = r'./dataset/' 
IMG_SIZE = 50
input_shape = (IMG_SIZE, IMG_SIZE, 1)
num_classes = 26
batch_size = 26
epochs = 12
training_data = []
CATEGORIES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def create_training_data(): # TAKE THE TRAINING AND TESTING IMAGES, FORMAT THEM CORRECTLY ADD THEM TO AN ARRAY
    for category in CATEGORIES:
        path = os.path.join(IMG_FOLDER, category)
        class_num = CATEGORIES.index(category)

        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])

def model(): # THE TRAINING MODEL AND PREDICTION OUTPUT

    X = []
    y = []

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X = X/255.0

    y = np.array(y)
    y = keras.utils.np_utils.to_categorical(y, num_classes)

    model = Sequential()

    model.add(Conv2D(28, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(56, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    model.summary()

    model.compile(loss = categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

    model.fit(X, y, batch_size = batch_size, epochs=epochs, verbose=1)
    score = model.evaluate(X, y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    for file in os.listdir('./predict/'):
        filename = os.fsdecode(file)
        prediction = model.predict([preparePred(filename)])
        pred_name = CATEGORIES[np.argmax(prediction)]
        print(filename, pred_name)
    
def preparePred(file):
    img_array = cv2.imread(os.path.join('./predict/', file), cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
  
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

def main():
    for category in CATEGORIES:
        path = os.path.join(IMG_FOLDER, category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))

    #new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

    create_training_data()
    random.shuffle(training_data)

    model()

main()