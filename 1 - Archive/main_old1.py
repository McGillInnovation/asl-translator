# The main code

import sys
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
paths = ['C:\\Users\\AanikaRahman\\Documents\\GitHub\\ASL3\\asl_dataset']
TOTAL_DATASET = 2515
x_train = []  # training lists
y_train = []
nb_classes = 36  # number of classes
img_rows, img_cols = 400, 400  # size of training images
img_channels = 3  # BGR channels
batch_size = 32
nb_epoch = 100  # iterations for training
data_augmentation = True

# dictionary for classes from char to numbers
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

# load the dataset and populate xtrain and ytrain
def load_data_set():
    for path in paths:
        for root, directories, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(".jpeg"):
                    #print("1")
                    #print(filename)
                    fullpath = os.path.join(root, filename)
                    #print("2")
                    #print(fullpath)
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
                    #break

sys.exit()

# create a model for training and return the model
def make_network(x_train):
    #Sequential model is a linear stack of layers.
    model = Sequential()
    #You can create a Sequential model by passing a list of layer instances to the constructor:
    print("Shape:")
    print(x_train.shape)
    # ???
    # 32 = number of convulation filters
    # (3, 3) = number of rows and columns in each convulation kernel
    # input_shape = (depth, width, height)
    # model.add(Convolution2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:])) 
    # OTHER
    #keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
    #2D convolution layer (e.g. spatial convolution over images).This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs. If  use_bias is True, a bias vector is created and added to the outputs. Finally, if activation is not None, it is applied to the outputs as well. When using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers, does not include the sample axis), e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in  data_format="channels_last".
    #TF Shape: (N_samples, height, width, N_channels)
    #model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=x_train.shape[1:]))
    #model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3L,400L,400L)))
    model.add(Convolution2D(32, (3, 3), padding='same', input_shape=(3,400,400))) 
    model.add(Activation('relu')) #don't know why there is a string inside the activation???
    model.add(Convolution2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3), padding='same')) #why is the number of filters doubled from 32?
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten()) #create one dimension
    #keras.layers.core.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
    #Just your regular densely-connected NN layer. Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True). Note: if the input to the layer has a rank greater than 2, then it is flattened prior to the initial dot product with kernel.
    model.add(Dense(512)) #why is this 512?
    model.add(Activation('relu'))
    model.add(Dense(nb_classes)) #is this the output layer?
    model.add(Activation('softmax'))

    return model

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
    Y_train = np_utils.to_categorical(y_train_new, nb_classes)

    #print("Shape:")
    #print((numpy.asarray(x_train)).shape)

    # run this if model is not saved.
    # UNCOMMENT LATER!!!
    model = make_network(numpy.asarray(x_train))
    #train_model(model,X_train,Y_train)
    #model.save('C:\\Users\\AanikaRahman\\Documents\\GitHub\\ASL3\\keras.model')

    # run this if model is already saved on disk.
    # model = keras.models.load_model('/home/snrao/IDE/PycharmProjects/ASL Finger Spelling Recognition/keras.model')

    #return model
    return Y_train # MUST CHANGE BACK!!!

model = trainData()

# Testing purposes
sys.exit()

####################################################################

# create a model for training and return the model
def make_network(x_train):
    #Sequential model is a linear stack of layers.
    model = Sequential()
    #You can create a Sequential model by passing a list of layer instances to the constructor:
    print("Shape:")
    print(x_train.shape)
    #keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
    #2D convolution layer (e.g. spatial convolution over images).This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs. If  use_bias is True, a bias vector is created and added to the outputs. Finally, if activation is not None, it is applied to the outputs as well. When using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers, does not include the sample axis), e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in  data_format="channels_last".
    #TF Shape: (N_samples, height, width, N_channels)
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=x_train.shape[1:])) 
    model.add(Activation('relu')) #don't know why there is a string inside the activation???
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same')) #why is the number of filters doubled from 32?
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten()) #create one dimension
    #keras.layers.core.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
    #Just your regular densely-connected NN layer. Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True). Note: if the input to the layer has a rank greater than 2, then it is flattened prior to the initial dot product with kernel.
    model.add(Dense(512)) #why is this 512?
    model.add(Activation('relu'))
    model.add(Dense(nb_classes)) #is this the output layer?
    model.add(Activation('softmax'))

    return model

# def trainData usede to be here!

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
              nb_epoch=nb_epoch)

    #doesn't this function need to return something? or will the other functions called return things while this one is running?
    




# called from main, when gesture is recognized. The gesture image is cropped and sent to this function.
def identifyGesture(handTrainImage):
    # saving the sent image for checking
    # cv2.imwrite("/home/snrao/IDE/PycharmProjects/ASL Finger Spelling Recognition/a0.jpeg", handTrainImage)

    # converting the image to same resolution as training data by padding to reach 1:1 aspect ration and then
    # resizing to 400 x 400. Same is done with training data in preprocess_image.py. Opencv image is first
    # converted to Pillow image to do this.
    
    #cv2.cvtColor: Converts an image from one color space to another.
    #http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
    handTrainImage = cv2.cvtColor(handTrainImage, cv2.COLOR_BGR2RGB)
    #The Image module provides a class with the same name which is used to represent a PIL image. 
    img = Image.fromarray(handTrainImage)
    img_w, img_h = img.size 
    #creating a background image based on the size of the maximum image
    M = max(img_w, img_h)
    background = Image.new('RGB', (M, M), (0, 0, 0))
    bg_w, bg_h = background.size
    offset = ((bg_w - img_w) / 2, (bg_h - img_h) / 2)
    background.paste(img, offset)
    size = 400,400
    background = background.resize(size, Image.ANTIALIAS)

    # saving the processed image for checking.
    # background.save("/home/snrao/IDE/PycharmProjects/ASL Finger Spelling Recognition/a.jpeg")

    # get image as numpy array and predict using model
    open_cv_image = numpy.array(background)
    background = open_cv_image.astype('float32')
    background = background / 255 #again why divded by 255. seems odd to hardcode something like this
    background = background.reshape((1,) + background.shape)
    predictions = model.predict_classes(background)

    # print predicted class and get the class name (character name) for the given class number and return it
    print predictions
    key = (key for key, value in classes.items() if value == predictions[0]).next() # i have no idea what this is
    return key


#It is used when a statement is required syntactically but you do not want any command or code to execute.
#The pass statement is a null operation; nothing happens when it executes. The pass is also useful in places where your code will eventually go, but has not been written yet (e.g., in stubs for example):
def nothing(x):
    pass


# Create a window to display the camera feed
cv2.namedWindow('Camera Output')
cv2.namedWindow('Hand')
cv2.namedWindow('HandTrain')

# TrackBars for fixing skin color of the person
cv2.createTrackbar('B for min', 'Camera Output', 0, 255, nothing)
cv2.createTrackbar('G for min', 'Camera Output', 0, 255, nothing)
cv2.createTrackbar('R for min', 'Camera Output', 0, 255, nothing)
cv2.createTrackbar('B for max', 'Camera Output', 0, 255, nothing)
cv2.createTrackbar('G for max', 'Camera Output', 0, 255, nothing)
cv2.createTrackbar('R for max', 'Camera Output', 0, 255, nothing)

# Default skin color values in natural lighting
# cv2.setTrackbarPos('B for min','Camera Output',52)
# cv2.setTrackbarPos('G for min','Camera Output',128)
# cv2.setTrackbarPos('R for min','Camera Output',0)
# cv2.setTrackbarPos('B for max','Camera Output',255)
# cv2.setTrackbarPos('G for max','Camera Output',140)
# cv2.setTrackbarPos('R for max','Camera Output',146)

# Default skin color values in indoor lighting
cv2.setTrackbarPos('B for min', 'Camera Output', 0)
cv2.setTrackbarPos('G for min', 'Camera Output', 130)
cv2.setTrackbarPos('R for min', 'Camera Output', 103)
cv2.setTrackbarPos('B for max', 'Camera Output', 255)
cv2.setTrackbarPos('G for max', 'Camera Output', 182)
cv2.setTrackbarPos('R for max', 'Camera Output', 130)

# Get pointer to video frames from primary device
videoFrame = cv2.VideoCapture(0) #0 is default camera

# Process the video frames
keyPressed = -1  # -1 indicates no key pressed. Can press any key to exit

# cascade xml file for detecting palm. Haar classifier
palm_cascade = cv2.CascadeClassifier('palm.xml')

# previous values of cropped variable
x_crop_prev, y_crop_prev, w_crop_prev, h_crop_prev = 0, 0, 0, 0

# previous cropped frame if we need to compare histograms of previous image with this to see the change.
# Not used but may need later.
_, prevHandImage = videoFrame.read()

# previous frame contour of hand. Used to compare with new contour to find if gesture has changed.
prevcnt = numpy.array([], dtype=numpy.int32)

# gesture static increments when gesture doesn't change till it reaches 10 (frames) and then resets to 0.
# gesture detected is set to 10 when gesture static reaches 10."Gesture Detected is displayed for next
# 10 frames till gestureDetected decrements to 0.
gestureStatic = 0
gestureDetected = 0

while keyPressed < 0:  # any key pressed has a value >= 0

    # Getting min and max colors for skin
    #uint8	Unsigned integer (0 to 255)
    min_YCrCb = numpy.array([cv2.getTrackbarPos('B for min', 'Camera Output'),
                             cv2.getTrackbarPos('G for min', 'Camera Output'),
                             cv2.getTrackbarPos('R for min', 'Camera Output')], numpy.uint8)
    max_YCrCb = numpy.array([cv2.getTrackbarPos('B for max', 'Camera Output'),
                             cv2.getTrackbarPos('G for max', 'Camera Output'),
                             cv2.getTrackbarPos('R for max', 'Camera Output')], numpy.uint8)

    # Grab video frame, Decode it and return next video frame
    # I think there is a typeo in success
    readSucsess, sourceImage = videoFrame.read()

    # Convert image to YCrCb
    imageYCrCb = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2YCR_CB)
    # Gaussian blur (also known as Gaussian smoothing) is the result of blurring an image by a Gaussian function. It is a widely used effect in graphics software, typically to reduce image noise and reduce detail.
    imageYCrCb = cv2.GaussianBlur(imageYCrCb, (5, 5), 0)

    # Find region with skin tone in YCrCb image
    skinRegion = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)

    # Do contour detection on skin region
    _, contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # sorting contours by area. Largest area first.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # get largest contour and compare with largest contour from previous frame.
    # set previous contour to this one after comparison.
    cnt = contours[0]
    #cv2.matchShapes() which enables us to compare two shapes, or two contours and returns a metric showing the similarity. The lower the result, the better match it is. It is calculated based on the hu-moment values. 
    ret = cv2.matchShapes(cnt, prevcnt, 2, 0.0)
    prevcnt = contours[0]

    # once we get contour, extract it without background into a new window called handTrainImage
    #numpy.zeros: Return a new array of given shape and type, filled with zeros.
    stencil = numpy.zeros(sourceImage.shape).astype(sourceImage.dtype)
    color = [255, 255, 255]
    #The function fillPoly fills an area bounded by several polygonal contours. The function can fill complex areas, for example, areas with holes, contours with self-intersections (some of their parts), and so forth.
    #http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html
    cv2.fillPoly(stencil, [cnt], color)
    #Calculates the per-element bit-wise conjunction of two arrays or an array and a scalar.
    #http://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html
    #bitwise conjunctions: http://virtualink.wikidot.com/fbtut:fbbitops
    handTrainImage = cv2.bitwise_and(sourceImage, stencil)

    # if comparison returns a high value (shapes are different), start gestureStatic over. Else increment it.
    if (ret > 0.70):
        gestureStatic = 0
    else:
        gestureStatic += 1

    # crop coordinates for hand.
    x_crop, y_crop, w_crop, h_crop = cv2.boundingRect(cnt)

    # place a rectange around the hand.
    cv2.rectangle(sourceImage, (x_crop, y_crop), (x_crop + w_crop, y_crop + h_crop), (0, 255, 0), 2)

    # if the crop area has changed drastically form previous frame, update it.
    if (abs(x_crop - x_crop_prev) > 50 or abs(y_crop - y_crop_prev) > 50 or
                abs(w_crop - w_crop_prev) > 50 or abs(h_crop - h_crop_prev) > 50):
        x_crop_prev = x_crop
        y_crop_prev = y_crop
        h_crop_prev = h_crop
        w_crop_prev = w_crop

    # create crop image
    handImage = sourceImage.copy()[max(0, y_crop_prev - 50):y_crop_prev + h_crop_prev + 50,
                max(0, x_crop_prev - 50):x_crop_prev + w_crop_prev + 50]

    # Training image with black background
    handTrainImage = handTrainImage[max(0, y_crop_prev - 15):y_crop_prev + h_crop_prev + 15,
                     max(0, x_crop_prev - 15):x_crop_prev + w_crop_prev + 15]

    # if gesture is static for 10 frames, set gestureDetected to 10 and display "gesture detected"
    # on screen for 10 frames.
    if gestureStatic == 10:
        gestureDetected = 10;
        print("Gesture Detected")
        letterDetected = identifyGesture(handTrainImage)  # todo: Ashish fill this function to return actual character

    if gestureDetected > 0:
        if (letterDetected != None):
            cv2.putText(sourceImage, letterDetected, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
        gestureDetected -= 1

    # Comparing histograms of this image and previous image to check if the gesture has changed.
    # Not accurate. So switched to contour comparisons.
    # hist1 = cv2.calcHist(handImage, [0, 1, 2], None, [8, 8, 8],
    #                     [0, 256, 0, 256, 0, 256])
    # hist1 = cv2.normalize(hist1,hist1).flatten()
    # hist2 = cv2.calcHist(prevHandImage, [0, 1, 2], None, [8, 8, 8],
    #                     [0, 256, 0, 256, 0, 256])
    # hist2 = cv2.normalize(hist2,hist2).flatten()
    # d = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    # # if d<0.9:
    # print(d)
    # prevHandImage = handImage

    # haar cascade classifier to detect palm and gestures. Not very accurate though.
    # Needs more training to become accurate.
    gray = cv2.cvtColor(handImage, cv2.COLOR_BGR2HSV)
    palm = palm_cascade.detectMultiScale(gray)
    for (x, y, w, h) in palm:
        cv2.rectangle(sourceImage, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # roi_gray = gray[y:y + h, x:x + w]
        roi_color = sourceImage[y:y + h, x:x + w]

    # to show convex hull in the image
    #Convex Hull will look similar to contour approximation, but it is not (Both may provide same results in some cases). Here, cv2.convexHull() function checks a curve for convexity defects and corrects it. Generally speaking, convex curves are the curves which are always bulged out, or at-least flat. And if it is bulged inside, it is called convexity defects. 
    #http://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)

    # counting defects in convex hull. To find center of palm. Center is average of defect points.
    count_defects = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        if count_defects == 0:
            center_of_palm = far
        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
        if angle <= 90:
            count_defects += 1
            if count_defects < 5:
                # cv2.circle(sourceImage, far, 5, [0, 0, 255], -1)
                center_of_palm = (far[0] + center_of_palm[0]) / 2, (far[1] + center_of_palm[1]) / 2
        cv2.line(sourceImage, start, end, [0, 255, 0], 2)
    # cv2.circle(sourceImage, avr, 10, [255, 255, 255], -1)


    # drawing the largest contour
    cv2.drawContours(sourceImage, contours, 0, (0, 255, 0), 1)

    # Display the source image and cropped image
    cv2.imshow('Camera Output', sourceImage)
    cv2.imshow('Hand', handImage)
    cv2.imshow('HandTrain', handTrainImage)

    # Check for user input to close program
    keyPressed = cv2.waitKey(30)  # wait 30 miliseconds in each iteration of while loop

# Close window and camera after exiting the while loop
cv2.destroyWindow('Camera Output')
videoFrame.release()
