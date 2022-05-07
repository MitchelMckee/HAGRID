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
from keras.datasets import mnist

import pickle
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt


training_data = []

IMG_FOLDER = r'./dataset/' 
IMG_SIZE = 50

CATEGORIES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
#, 'C_A', 'C_B', 'C_C', 'C_D', 'C_E', 'C_F', 'C_G', 'C_H', 'C_I', 'C_J', 'C_K', 'C_L', 'C_M', 'C_N', 'C_O', 'C_P', 'C_Q', 'C_R', 'C_S', 'C_T', 'C_U', 'C_V', 'C_W', 'C_X', 'C_Y', 'C_Z']
#CATEGORIES = ['a', 'b']

input_shape = (IMG_SIZE, IMG_SIZE, 1)
num_classes = 26
batch_size = 26
epochs = 10

for category in CATEGORIES:
    path = os.path.join(IMG_FOLDER, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array)
        break
    break
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) # CHANGE ALL IMAGES TO IMG_SIZE

def create_training_data():
    for category in CATEGORIES:
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            plt.imshow(new_array)
           # plt.show()
            training_data.append([new_array, class_num])
           
    return training_data

create_training_data()
random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)


print(X[1])
print(y[1])


#x_test = X[:round(len(training_data)/2)]
#x_train = X[round(len(training_data)/2):]
#y_train = y[:round(len(training_data)/2)]
#y_test = y[round(len(training_data)/2):]

#x_train = x_train / 255
#x_test = x_test / 255 

#x_train = x_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
#x_test = x_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

'''
print('-----------------------------------------')
print(x_train.shape, 'x_train shape')
print(x_test.shape, 'x_test shape')
print(x_train[0], 'x_train sample')
print(x_test[0], 'x_test sample')
print('-----------------------------------------')
print(y_train.shape, 'y_train shape')
print(y_test.shape, 'y_test shape')
print(y_train[0], 'y_train sample\n')
print(y_test[0], 'y_test sample')
print('-----------------------------------------')
'''
####################################################################################################################################################

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.9))
model.add(Flatten()) # This line is to convert from matrices to vectors
model.add(Dense(26, activation='relu'))
model.add(Dense(num_classes, activation='softmax')) # we are working with vectors now, so we use a Dense layer instead of Conv2d
model.summary()


model.compile(loss=categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y,
          batch_size=batch_size, validation_split=0.1)
score = model.evaluate(X, y, verbose=0)


print(np.argmax(model.predict(X[:1]), axis=-1))
print('Test loss:', score[0])
print('Test accuracy:', score[1])
