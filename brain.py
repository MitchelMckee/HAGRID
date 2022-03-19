from __future__ import print_function
from random import triangular
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras import backend as K

from datasetImport import create_training_data
training_data = create_training_data()


num_classes = 26
batch_size = 128
epochs = 10

x_train = training_data
#x_test = training_data[:1500, :, :]

print(training_data)