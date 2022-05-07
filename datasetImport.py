import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt

#GLOBAL
global training_data
training_data = []

#DATASET PARAMS
IMG_FOLDER = r'./dataset/' 
IMG_SIZE = 50
CATEGORIES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

#CREATE THE IMG_ARRAY
for category in CATEGORIES:
    path = os.path.join(IMG_FOLDER, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        plt.imshow(img_array)
        break
    break

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) # CHANGE ALL IMAGES TO 50 x 50
plt.imshow(new_array)
#plt.show()

#PREPARE ALL IMAGES
def create_training_data(training_data):
    for category in CATEGORIES:
        path = os.path.join(IMG_FOLDER, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img))
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])
            
      
    random.shuffle(training_data)
    #print(len(training_data))    
    return training_data

#SPLIT THE IMAGE DATA INTO LABELS AND FEATURES
def labelFeatures(training_data):
    
    create_training_data(training_data)
    X = []
    y = []

    for features, label in training_data:
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE ,1)
    return X, y


#pickle_out = open('X.pickle', 'wb')   
#pickle.dump(X, pickle_out)
#pickle_out.close()

#pickle_out = open('y.pickle', 'wb')
#pickle.dump(y, pickle_out)
#pickle_out.close()

labelFeatures(training_data)

