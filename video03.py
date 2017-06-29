import cv2
#import numpy
#vidcap = cv2.VideoCapture('big_buck_bunny_720p_5mb.mp4')
#C:/Users/otterrafting/Documents/GitHub/SensAI/output/signs1/vgg16/classifier_(0.1).cpickle

import os
import sys
#The traceback module works with the call stack to produce error messages. 
import traceback

import cPickle as pickle
#import cloudpickle #as pickle
from sklearn.externals import joblib
#import joblib

#having difficulty figuring out what the common package does
from common.config import get_config
from common.image_transformation import apply_image_transformation
from common.image_transformation import resize_image





vidcap = cv2.VideoCapture(0)

success,frame = vidcap.read()
count = 0
success = True
while success:
  success,frame = vidcap.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Display the resulting frame
  cv2.imshow('frame',gray)
  print 'Read a new frame: ', success
  

  if not success:
            print("Failed to capture image!")
            continue
  frame = resize_image(frame, 400)
  cv2.imshow("Webcam recording", frame)
        
  try:
        #frame = apply_image_transformation(frame)
        frame_flattened = frame.flatten() #numpy - collapses array into one dimension
        #print(model_serialized_path)
        #model_serialized_path = "C:\\Users\\AanikaRahman\\Documents\\GitHub\\ASL1\\Sign-Language-Recognition\\data\\generated\\output\\svm\\model-serialized-svm.pkl"
        model_serialized_path = "C:/Users/otterrafting/Documents/GitHub/SensAI/output/signs1/vgg16/classifier_(0.1).cpickle"
        #classifier_model = joblib.load(model_serialized_path)
        #classifier_model = pickle.load(open(model_serialized_path),encoding='latin1')
        classifier_model = pickle.load(open(model_serialized_path,"rb"))
        predicted_labels = classifier_model.predict(frame_flattened)
        predicted_label = predicted_labels[0]
        print("Predicted label = {}".format(predicted_label))
        predicted_image = get_image_from_label(predicted_label) #unsure why predicted image has its value set twice. i guess images have multiple elements to them?
        predicted_image = resize_image(predicted_image, 200)
        cv2.imshow("Prediction = '{}'".format(predicted_label), predicted_image)
  except Exception:
        exception_traceback = traceback.format_exc()
        print("Error while applying image transformation with the following exception trace:\n{}".format(
                exception_traceback))	
  
  cv2.imwrite("frame%d.jpg" % count, frame)     # save frame as JPEG file
  count += 1

  if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# When everything done, release the capture
vidcap.release()
cv2.destroyAllWindows()