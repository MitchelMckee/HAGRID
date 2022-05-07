import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator


training_data = []

IMG_FOLDER = r'./dataset/' 
IMG_SIZE = 50
#CATEGORIES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
CATEGORIES = ['a', 'b']

for category in CATEGORIES:
    path = os.path.join(IMG_FOLDER, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        break
    break
print(img_array.shape)

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

training_data = []

def create_training_data():
    for category in CATEGORIES:
        class_num = CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):
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
y = np.array(y)

