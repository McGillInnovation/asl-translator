#!/usr/bin/python
import os
import sys
import cv2
import numpy as np
import math
import glob
import h5py
import json
import cPickle
import datetime
import time
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
# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# load the user configs
with open('conf\\conf_signs1&2_vgg16.json') as f:    
	config = json.load(f)

# config variables
model_name = config["model"]
weights = config["weights"]
include_top = config["include_top"]
train_path = config["train_path"]
model_path = config["model_path"]
#weights_path = config["weights_path"]

# start time
print ("[STATUS] start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
start = time.time()

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

# path to training dataset
train_labels = os.listdir(train_path)

# encode the labels
print ("[INFO] encoding labels...")
le = LabelEncoder() # ???
le.fit([tl for tl in train_labels])

# variables to hold features and labels
features = []
labels   = []

# loop over all the labels in the folder (create dictionary of classes)
for (i, label) in enumerate(train_labels):
	#print ("0 --- {}".format((i, label)))
	cur_path = train_path + "\\" + label
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

# encode the labels using LabelEncoder
targetNames = np.unique(labels) # ???
le = LabelEncoder() # ???
le_labels = le.fit_transform(labels) # ???

# get the shape of training labels
print ("[STATUS] training labels: {}".format(le_labels))
print ("[STATUS] training labels shape: {}".format(le_labels.shape))

# save features and labels
h5f_data = h5py.File(model_path + "\\features.h5", 'w')
h5f_data.create_dataset('dataset_1', data=np.array(features))

h5f_label = h5py.File(model_path + "\\labels.h5", 'w')
h5f_label.create_dataset('dataset_1', data=np.array(le_labels))

h5f_data.close()
h5f_label.close()

# save model and weights
model_json = model.to_json()
with open(model_path + "\\model.json", "w") as json_file:
	json_file.write(model_json)

# save weights
model.save_weights(model_path + "\\model.h5")
print("[STATUS] saved model and weights to disk..")
print("[STATUS] features and labels saved..")

# end time
end = time.time()
print ("[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))