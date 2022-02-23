#%% Imports

# Built-in imports
import os, datetime, pathlib, random, time
from glob import glob
import math
from math import ceil, floor

# Basic imports
import numpy as np
import pandas as pd   

# Ploting libraries
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10,10)
mpl.rcParams['axes.grid'] = False
%matplotlib inline

import seaborn as sns
sns.set(color_codes=True)

import PIL
from PIL import Image
import skimage
from skimage import color, io
#from skimage import data
from skimage import transform

# Tensorflow imports
import tensorflow as tf
print(f'Tensorflow version is: {tf.version.VERSION}')
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print('GPU device not found')
elif device_name == '/device:GPU:0':
  print('Found GPU at: {}'.format(device_name))

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image as kp_image
from tensorflow.keras.callbacks import TensorBoard 
from tensorflow.keras import applications, optimizers, models, losses, layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
#from tensorflow.keras.metrics import 

# Train-test-split and evaluation imports
from sklearn.model_selection import train_test_split # If required
from sklearn import metrics # classification report
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold

#%% Separate data
# Get cat and dog images
path = './raw_data/training_data'
data = glob(path + '/*.*')
cats = [f for f in data if 'cat' in f]
dogs = [f for f in data if 'dog' in f]

label_encoded = {1:'dog',2:'cat'}
labels = [1 if 'dog' in f else 0 for f in data]

# print out the number of images
print(
    f'Number of cats: {len(cats)}\n',
    f'Number of dogs: {len(dogs)}'
)

# %% Check data 

# Baseline Model
# A good starting point is the general architectural principles of the VGG models. 
# The architecture involves stacking convolutional layers with small 3×3 filters followed by a max pooling layer. 
# Together, these layers form a block, and these blocks can be repeated where the number of filters in each block is increased with the depth of the network such as 32, 64, 128, 256 for the first four blocks of the model. 
# Padding is used on the convolutional layers to ensure the height and width shapes of the output feature maps matches the inputs.

def define_baseline_model(img_array,label_array, dropout_ratio):
    num_classes = train_label.shape[1]
    image_size = train_data.shape[1]

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (5, 5), padding = 'same', input_shape = (image_size, image_size, 1)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(dropout_ratio))

    model.add(tf.keras.layers.Conv2D(64, (5, 5), padding = 'same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(dropout_ratio))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding = 'same'))
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(num_classes))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Activation('softmax'))
	# compile model
    opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model