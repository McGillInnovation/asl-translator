import cv2
import numpy
import math
import os
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from PIL import Image
import keras

# path to the dataset
paths = ['/home/snrao/IDE/PycharmProjects/ASL Finger Spelling Recognition/asl_dataset']
TOTAL_DATASET = 2515
x_train = []  # training lists
y_train = []
nb_classes = 36  # number of classes
img_rows, img_cols = 400, 400  # size of training images
img_channels = 3  # BGR channels
batch_size = 32
nb_epoch = 100  # iterations for training
data_augmentation = True