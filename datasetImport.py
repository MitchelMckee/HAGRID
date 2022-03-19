import numpy as np
import cv2
import os
import pickle 
import random
import matplotlib.pyplot as plt

#DATASET PARAMS
IMG_FOLDER = r'./dataset/' 
IMG_SIZE = 50
CATEGORIES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

for category in CATEGORIES:
    path = os.path.join(IMG_FOLDER, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        plt.imshow(img_array)
        break
    break

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array)
#plt.show()


def create_training_data():
    training_data = np.array([])
    for i in range(len(CATEGORIES)):
        category = CATEGORIES[i]
        path = os.path.join(IMG_FOLDER, category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img))
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
           #training_data.append([new_array, i])
            training_data = np.append(training_data, [new_array, i])
            #print(training_data)
    random.shuffle(training_data)
    


    return training_data


#print(training_data)
"""
X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open('X.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()
"""


