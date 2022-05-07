from __future__ import print_function

import keras
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.losses import categorical_crossentropy
from keras import backend as K
from keras.utils import np_utils

import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt

IMG_FOLDER = r'./dataset/' 
IMG_SIZE = 50

input_shape = (IMG_SIZE, IMG_SIZE, 1)
num_classes = 26
batch_size = 26
epochs = 30

training_data = []

CATEGORIES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

for category in CATEGORIES:
    path = os.path.join(IMG_FOLDER, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
 
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(IMG_FOLDER, category)
        class_num = CATEGORIES.index(category)

        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])

create_training_data()
random.shuffle(training_data)

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
model.add(Conv2D(56, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(100, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))

model.summary()

model.compile(loss = categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

#model.fit(X, y, batch_size = batch_size, epochs=epochs, verbose=1)
history = model.fit(X, y, batch_size = batch_size, epochs=epochs, verbose=1)
score = model.evaluate(X, y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()