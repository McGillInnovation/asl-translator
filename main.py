#!/usr/bin/python
import sys
import cv2
import numpy
import math
import os
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from PIL import Image
import keras

# KEY/ANNOTATIONS
# EDIT! ~ edit
# [?????] or ? ~ question


# Initialising 
paths = ['C:\\Users\\AanikaRahman\\Documents\\GitHub\\ASL3\\asl_dataset'] # path to the dataset
TOTAL_DATASET = 2515 # EDIT!
x_train = [] # training lists
y_train = []
n_classes = 36 # number of classes
img_rows, img_cols = 400, 400 # size of training images
img_channels = 3 # BGR channels
batch_size = 32
n_epoch = 100 # iterations for training
data_augmentation = True

# dictionary for classes from char to numbers
# EDIT: for loop, mapping folder names to 0 to (number of folders)-1
classes = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'a': 10,
    'b': 11,
    'c': 12,
    'd': 13,
    'e': 14,
    'f': 15,
    'g': 16,
    'h': 17,
    'i': 18,
    'j': 19,
    'k': 20,
    'l': 21,
    'm': 22,
    'n': 23,
    'o': 24,
    'p': 25,
    'q': 26,
    'r': 27,
    's': 28,
    't': 29,
    'u': 30,
    'v': 31,
    'w': 32,
    'x': 33,
    'y': 34,
    'z': 35,
}


# Load dataset and populate x_train and y_train
def load_data_set():
    for path in paths:
        for root, directories, filenames in os.walk(path): # EDIT!
            for filename in filenames:
                if filename.endswith(".jpeg"):
                    fullpath = os.path.join(root, filename)
                    img = load_img(fullpath)
                    #print("3")
                    #print(img)
                    img = img_to_array(img)
                    #print("4")
                    #print(img)
                    x_train.append(img)
                    #print("5")
                    #print(x_train)
                    t = fullpath.rindex('\\') # last index of string before '\\'
                    #print("6")
                    #print(t)
                    fullpath = fullpath[0:t]
                    #print("7")
                    #print(fullpath)
                    n = fullpath.rindex('\\') # last index of string
                    #print("8a")
                    #print(n)
                    #print("8b")
                    #print(fullpath[n + 1:t])
                    y_train.append(classes[fullpath[n + 1:t]])
                    #print("9")
                    #print(y_train)


# Create a model for training and return the model
def make_network(x_train):
    # Sequential model is a linear stack of layers, created by passing list of layer instances to constructor
    model = Sequential()
    # x_train.shape = (N_samples, height, width, N_channels) ~ i.e. (N_samples, 400, 400, 3)
    print("Shape:")
    print(x_train.shape)
    # https://keras.io/getting-started/sequential-model-guide/ --- Note "VGG-like convnet"
    # https://faroit.github.io/keras-docs/0.2.0/examples/ --- Note "VGG-like convnet" & "learning image captions..."
    # https://elitedatascience.com/keras-tutorial-deep-learning-in-python --- Note final code
    # input 400x400 RGB images with 3 channels, apply 32 convolution filters of size 3x3 each
    # https://keras.io/layers/convolutional/ --- Note "Conv2D" arguments
    # filters (number output of filters in the convolution) = 32
    # kernel_size (width & height of 2D convolution window) = (3, 3)
    # input_shape = (depth, width, height) ~ i.e. (400, 400, 3)
    # data_format = channels_last (default)
    # PREVIOUSLY (before converting to Keras 2 format):
    # model.add(Convolution2D(32, (3, 3), padding='same', input_shape=(400,400,3)))
    # model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=x_train.shape[1:]))
    model.add(Convolution2D(32, (3, 3), padding='valid', activation='relu', input_shape=x_train.shape[1:]))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64, (3, 3), padding='valid', activation='relu')) # why number of filters doubled from 32?
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # https://keras.io/layers/core --- Note "Dense" & "Flatten"
    # Dense: regular densely-connected NN layer, where output = activation(dot(input, kernel) + bias)
    # activation (element-wise activation function passed as the activation argument)
    # kernel (weights matrix created by the layer)
    # bias (bias vector created by the layer)
    # input: if rank >2, flattened prior to initial dot product with kernel
    # Flatten: convert input shape from 4D to 1D [?????]
    # before ~ model.output_shape = (None, 64, 400, 400)
    # after ~ model.output_shape = (None, 64*400*400=10240000) [?????]
    model.add(Flatten())
    model.add(Dense(512, activation='relu')) # why is this 512 [?????]
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax')) # output layer [?????]
    return model


# training model which was created
def train_model(model, X_train, Y_train):
    # let's train the model using SGD (Stochastic gradient descent) + momentum (how original).
    #keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    #lr: float >= 0. Learning rate.
    #momentum: float >= 0. Parameter updates momentum.
    #decay: float >= 0. Learning rate decay over each update.
    #nesterov: boolean. Whether to apply Nesterov momentum.
    #http://cs231n.github.io/neural-networks-3/#sgd
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #compile(self, optimizer, loss, metrics=None, sample_weight_mode=None)
    #what is self?!?!?!?
    #An optimizer. This could be the string identifier of an existing optimizer (such as rmsprop or adagrad), or an instance of the     Optimizer class. See: optimizers.
    #A loss function. This is the objective that the model will try to minimize. It can be the string identifier of an existing loss function (such as categorical_crossentropy or mse), or it can be an objective function. See: losses.
    #A list of metrics. For any classification problem you will want to set this to metrics=['accuracy']. A metric could be the string identifier of an existing metric or a custom metric function.
    #sample_weight_mode: if you need to do timestep-wise sample weighting (2D weights), set this to "temporal". "None" defaults to sample-wise weights (1D).
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    #fit(self, x, y, batch_size=32, epochs=10, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
    #further documentation at : https://keras.io/models/sequential/
    # Check if its following model mentioned by Andrew Piper
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=n_epoch)

    #doesn't this function need to return something? or will the other functions called return things while this one is running?


# loads data set, converts the trianing arrays into required formats of numpy arrays and calls make_network to
# create a model and then calls train_model to train it and then saves the model in disk. OR just loads the model
# from disk.
def trainData():
    load_data_set()
    # Convert list to 1D array of each sample's class
    a = numpy.asarray(y_train)
    # Convert 1D array to nested array (array of arrays), where each array contains class for each sample
    y_train_new = a.reshape(a.shape[0], 1)
    # Noting x_train from load_data_set():
    # len(x_train) = number of samples
    # each sample image is composed of 400*400 pixels:
    # x_train[0].size = 480000 (i.e. 400*400*3) 
    # x_train[0][0].size = 1200 (i.e. 400*3)
    # x_train[0][0][0].size = 3 (for each pixel)
    # Convert X_train from list to array of floats (otherwise same as above size-wise)
    # Therefore X_train.size = (number of samples)*400*400*3
    # Convert to float32 to be able to divide after (???)
    # Properties mentioned for x_train (e.g. "dtype=float32") removed for X_train
    # https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.ndarray.astype.html
    # ndarray.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)
    # All floats are between 0 and 255 (RGB value scale)
    X_train = numpy.asarray(x_train).astype('float32')
    # All floats are between 0 and 1 (unit RBG value scale)
    # BUT why divide by 255 (???) 
    X_train = X_train / 255.0 
    # to_categorical(y, num_classes=None)
    # Y_train: an array of 'binary' arrays; each 'binary' array has a '1' at array[class] and '0' at all other indexes 
    # Y_train: visualise as rows=samples and columns=classes (coverting class vector to 'binary class matrix')
    # Y_train.size = number of classes * number of samples (i.e. approx 36*(70*26))
    # Y_train[0].size = number of classes (i.e. 36)
    Y_train = np_utils.to_categorical(y_train_new, n_classes)

    #print("Shape:")
    #print((numpy.asarray(x_train)).shape)

    # run this if model is not saved.
    # UNCOMMENT LATER!!!
    model = make_network(numpy.asarray(x_train))
    train_model(model,X_train,Y_train)
    #model.save('C:\\Users\\AanikaRahman\\Documents\\GitHub\\ASL3\\keras.model')

    # run this if model is already saved on disk.
    # model = keras.models.load_model('/home/snrao/IDE/PycharmProjects/ASL Finger Spelling Recognition/keras.model')

    #return model
    return Y_train # MUST CHANGE BACK!!!


model = trainData()