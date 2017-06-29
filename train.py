#!/usr/bin/python
import os
import sys
import numpy as np
import h5py
import json
import cPickle
import seaborn as sns
import matplotlib.pyplot as plt
# sklearn imports
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
#from sklearn import metrics
#from sklearn import linear_model
#from sklearn import svm
#from sklearn.model_selection import train_test_split
#from sklearn.neighbors import KNeighborsClassifier


# load the user configs
with open('conf/conf_signs1&2_vgg16.json') as f:    
	config = json.load(f)

# config variables
train_path = config["train_path"]
model_path = config["model_path"]
test_size = config["test_size"]
seed = config["seed"]
num_classes = config["num_classes"]

# import features and labels
h5f_data = h5py.File(model_path + "\\features.h5", 'r')
h5f_label = h5py.File(model_path + "\\labels.h5", 'r')

features_string = h5f_data['dataset_1'] #???
labels_string = h5f_label['dataset_1'] #???

features = np.array(features_string)
labels = np.array(labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of features and labels
print "[INFO] features shape: {}".format(features.shape)
print "[INFO] labels shape: {}".format(labels.shape)

print "[INFO] training started..."
# split the training and testing data
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(features),
                                                                  np.array(labels),
                                                                  test_size=test_size,
                                                                  random_state=seed) #EXPLAIN

print "[INFO] splitted train and test data..."
print "[INFO] test size   : {}".format(test_size)
print "[INFO] train data  : {}".format(trainData.shape)
print "[INFO] test data   : {}".format(testData.shape)
print "[INFO] train labels: {}".format(trainLabels.shape)
print "[INFO] test labels : {}".format(testLabels.shape)

# use logistic regression as the model
print("[INFO] creating model...")
model = LogisticRegression(random_state=seed) #EXPLAIN
model.fit(trainData, trainLabels)

# use rank-1 and rank-5 predictions
print("[INFO] evaluating model...")
f = open(model_path + "\\results_(" + str(test_size) + ").txt", "w")
rank_1 = 0 #EXPLAIN
rank_5 = 0 #EXPLAIN

# loop over test data
for (label, features) in zip(testLabels, testData):
	# predict the probability of each class label and
	# take the top-5 class labels

	predictions = model.predict_proba(np.atleast_2d(features))[0] #EXPLAIN
	#print ("shape1")
	#print (predictions)
	predictions = np.argsort(predictions)[::-1][:5]
	#print ("shape2")
	#print (predictions)
	# rank-1 prediction increment
	if label == predictions[0]: # comparing most likely prediction?
		rank_1 += 1

	# rank-5 prediction increment
	if label in predictions: # comparing all predictions?
		rank_5 += 1

# convert accuracies to percentages
rank_1 = (rank_1 / float(len(testLabels))) * 100
rank_5 = (rank_5 / float(len(testLabels))) * 100

# write the accuracies to file
f.write("Rank-1: {:.2f}%\n".format(rank_1))
f.write("Rank-5: {:.2f}%\n\n".format(rank_5))

# evaluate the model of test data
preds = model.predict(testData) #EXPLAIN

# ...
print "[INFO] train score : {}".format(model.score(trainData, trainLabels))
print "[INFO] test score  : {}".format(model.score(testData, testLabels))

# write the classification report to file
f.write("{}\n".format(classification_report(testLabels, preds)))
f.close()

# dump classifier to file
print("[INFO] saving model...")
f = open(model_path + "\\classifier_(" + str(test_size) + ").cpickle", "w")
f.write(cPickle.dumps(model))
f.close()

# display the confusion matrix
print ("[INFO] confusion matrix")

# get the list of training lables
# maybe to label axis in matrix later, but not needed now
# labels = sorted(list(os.listdir(train_path)))

# plot the confusion matrix
cm = confusion_matrix(testLabels, preds)
sns.heatmap(cm, annot=True, cmap="Set2")
#plt.show()
plt.savefig(model_path + "\\matrix_(" + str(test_size) + ").png")