import cv2
#import numpy
#vidcap = cv2.VideoCapture('big_buck_bunny_720p_5mb.mp4')
#C:/Users/otterrafting/Documents/GitHub/SensAI/output/signs1/vgg16/classifier_(0.1).cpickle

import os
import sys
import json
#The traceback module works with the call stack to produce error messages. 
import traceback
import datetime
import numpy as np

import pickle
import cPickle
#import cloudpickle #as pickle
from sklearn.externals import joblib
#import joblib

# sklearn imports
from sklearn.preprocessing import LabelEncoder
# keras imports
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json

#having difficulty figuring out what the common package does
from common.config import get_config
from common.image_transformation import apply_image_transformation
from common.image_transformation import resize_image

# load the user configs
with open('conf/conf_signs1&2_vgg16.json') as f:    
  config = json.load(f)

# config variables
model_name = config["model"]
weights = config["weights"]
include_top = config["include_top"]
frames_path = config["frames_path"]
model_path = config["model_path"]
test_size = config["test_size"]
seed = config["seed"]
num_classes = config["num_classes"]

# start time
#print ("[STATUS] start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
#start = time.time()

# create the pretrained models
# check for pretrained weight usage or not
# check for top layers to be included or not
if model_name == "vgg16":
  base_model = VGG16(weights=weights)
  #model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
  model = Model(inputs=base_model.inputs, outputs=base_model.get_layer('fc1').output)
  image_size = (224, 224)
elif model_name == "vgg19":
  base_model = VGG19(weights=weights)
  model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
  image_size = (224, 224)
elif model_name == "resnet50":
  base_model = ResNet50(weights=weights)
  model = Model(input=base_model.input, output=base_model.get_layer('flatten').output)
  image_size = (224, 224)
elif model_name == "inceptionv3":
  base_model = InceptionV3(weights=weights)
  model = Model(input=base_model.input, output=base_model.get_layer('flatten').output)
  image_size = (299, 299)
elif model_name == "xception":
  base_model = Xception(weights=weights)
  model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
  image_size = (299, 299)
else:
  base_model = None

print ("[INFO] successfully loaded base model and model...")

vidcap = cv2.VideoCapture(1)
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

  # save frames
  cv2.imwrite("frames\\random\\frame%d.jpg" % count, frame)     # save frame as JPEG file
  count += 1

  # PROCESS:
  # extract features
  # make predictions on features
  # output

  if cv2.waitKey(500) & 0xFF == ord('q'):
    break

# When everything done, release the capture
vidcap.release()
cv2.destroyAllWindows()

# path to training dataset
frames_labels = os.listdir(frames_path)

# encode the labels
print ("[INFO] encoding labels...")
le = LabelEncoder() # ???
le.fit([tl for tl in frames_labels])

# variables to hold features and labels
features = []
labels   = []

# loop over all the labels in the folder (create dictionary of classes)
for (i, label) in enumerate(frames_labels):
  #print ("0 --- {}".format((i, label)))
  cur_path = frames_path + "\\" + label
  #print ("0 --- {}".format(cur_path))
  #for image_path in glob.glob(cur_path + "\\*.jpg"):
  for root, directories, filenames in os.walk(cur_path):
    for image_file in filenames:
      if image_file.endswith(".jpeg") or image_file.endswith(".jpg"):
        image_path = os.path.join(cur_path, image_file)
        img = image.load_img(image_path, target_size=image_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) # ???
        x = preprocess_input(x)
        feature = model.predict(x)
        flat = feature.flatten()
        features.append(flat)
        labels.append(label)
        print ("[INFO] processed - {} - {}".format(i, image_file))
  print ("[INFO] completed label - {}".format(label))

try:
  #frame = apply_image_transformation(frame)
  #print("0")
  #print(frame.shape)
  #frame_flattened = frame.flatten() #numpy - collapses array into one dimension
  #print(model_serialized_path)
  #model_serialized_path = "C:\\Users\\AanikaRahman\\Documents\\GitHub\\ASL1\\Sign-Language-Recognition\\data\\generated\\output\\svm\\model-serialized-svm.pkl"
  model_serialized_path = "output\\trial1\\vgg16\\classifier.cpickle" 
  #classifier_model = joblib.load(model_serialized_path)
  #classifier_model = pickle.load(open(model_serialized_path),encoding='latin1')
  classifier_model = cPickle.load(open(model_serialized_path,"r"))
  #print("1")
  #print(frame_flattened.shape)
  predicted_labels = classifier_model.predict(features) #convert to numpy array?
  predicted_label = predicted_labels[0]
  print("Predicted labels = {}".format(predicted_labels))
  #predicted_image = get_image_from_label(predicted_label) #unsure why predicted image has its value set twice. i guess images have multiple elements to them?
  #predicted_image = resize_image(predicted_image, 200)
  #cv2.imshow("Prediction = '{}'".format(predicted_label), predicted_image)
except Exception:
  exception_traceback = traceback.format_exc()
  print("Error while applying image transformation with the following exception trace:\n{}".format(exception_traceback)) 







sys.exit()

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
        print("0")
        print(frame.shape)
        frame_flattened = frame.flatten() #numpy - collapses array into one dimension
        #print(model_serialized_path)
        #model_serialized_path = "C:\\Users\\AanikaRahman\\Documents\\GitHub\\ASL1\\Sign-Language-Recognition\\data\\generated\\output\\svm\\model-serialized-svm.pkl"
        model_serialized_path = "output\\signs1\\vgg16\\classifier_(0.1).cpickle" 
        #classifier_model = joblib.load(model_serialized_path)
        #classifier_model = pickle.load(open(model_serialized_path),encoding='latin1')
        #classifier_model = cPickle.load(open(model_serialized_path,"r"))
        classifier_model = cPickle.load(open(model_serialized_path,"rb"))
        print("1")
        print(frame_flattened.shape)
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
  
  cv2.imwrite("frames\\frame%d.jpg" % count, frame)     # save frame as JPEG file
  count += 1

  if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# When everything done, release the capture
vidcap.release()
cv2.destroyAllWindows()