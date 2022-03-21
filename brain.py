from __future__ import print_function
from random import triangular
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras import backend as K

import numpy as np
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt

training_data = []

IMG_FOLDER = r'./dataset/' 
IMG_SIZE = 50

#CATEGORIES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
CATEGORIES = ['a', 'b']

input_shape = (IMG_SIZE, IMG_SIZE, 1)
num_classes = 26
batch_size = 128
epochs = 10

#x_test = training_data[:round(len(training_data)/2)]
#x_train = training_data[round(len(training_data)/2):]

for category in CATEGORIES:
    path = os.path.join(IMG_FOLDER, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array)
        break
    break

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) # CHANGE ALL IMAGES TO 50 x 50

def create_training_data():
    for category in CATEGORIES:
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])
           
      

    return training_data

random.shuffle(training_data)
create_training_data()

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE ,1)
y = np.array(y)

X = X/255.0




####################################################################################################################################################

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.3))
model.add(Flatten()) # This line is to convert from matrices to vectors
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax')) # we are working with vectors now, so we use a Dense layer instead of Conv2d
model.summary()
model.compile(loss=categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X, y))
score = model.evaluate(X, y, verbose=0)