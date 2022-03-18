from __future__ import print_function
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

#SAVE MODEL
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib
import pickle

#PARAMS
letter_classes = 26
batch_size = 128
epochs = 5

#DATASET PARAMS
img_folder = '/Users/mitchelmckee/Desktop/HAGRID/dataset/img'
img_width, img_height = 28, 28

def create_dataset(img_folder):
   
    img_data_array=[]
    class_name=[]
   
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
       
            image_path= os.path.join(img_folder, dir1,  file)
            image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
            image=cv2.resize(image, (img_height, img_width),interpolation = cv2.INTER_AREA)
            image=np.array(image)
            image = image.astype('float32')
            image /= 255 
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name
   

img_data, class_name = create_dataset(r'/Users/mitchelmckee/Desktop/HAGRID/dataset/img_subset')

target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
print(target_dict)

#(x_train, y_train), (x_test, y_test) = 