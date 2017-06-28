#!/usr/bin/env python2
#The OS module in Python provides a way of using operating system dependent functionality. 
#The functions that the OS module provides allows you to interface with the
#underlying operating system that Python is running on – be that Windows, Mac or Linux. 
import os
#An interpreter is a program that reads and executes code. This includes source code, pre-compiled code, and scripts. 
#The sys module provides information about constants, functions and methods of the Python interpreter. 
#dir(system) gives a summary of the available constants, functions and methods. 
#Another possibility is the help() function. Using help(sys) provides valuable detail information. 
#The module sys informs e.g. about the maximal recursion depth (sys.getrecursionlimit() ) 
#and provides the possibility to change (sys.setrecursionlimit()) 
import sys
#The csv module implements classes to read and write tabular data in CSV format. 
#It allows programmers to say, “write this data in the format preferred by Excel,”
import csv

#NumPy is the fundamental package for scientific computing with Python. It contains among other things:
#a powerful N-dimensional array object
#sophisticated (broadcasting) functions
#tools for integrating C/C++ and Fortran code
#useful linear algebra, Fourier transform, and random number capabilities
#Besides its obvious scientific uses, NumPy can also be used as an efficient multi-dimensional container of generic data. 
#Arbitrary data-types can be defined. This allows NumPy to seamlessly and speedily integrate with a wide variety of databases.
import numpy as np
#scikit learn (sklearn) A set of python modules for machine learning and data mining
from sklearn import metrics
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import train_test_split
#In the specific case of the scikit, it may be more interesting to use 
#joblib’s replacement of pickle (joblib.dump & joblib.load), 
#which is more efficient on objects that carry large numpy arrays internally as is often the case for fitted scikit-learn estimators, 
#but can only pickle to the disk and not to a string:
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier


from common.config import get_config


def print_with_precision(num):
    return "%0.5f" % num


def read_images_transformed(images_transformed_path):
    print("\nReading the transformed images file located at path '{}'...".format(
        images_transformed_path))

    images = [] #make array?
    labels = [] #make array?
    with open(images_transformed_path) as images_transformed_file:
        reader = csv.reader(images_transformed_file, delimiter=',')
        for line in reader: #where did the variable line come from? is this where it is initialized?
            label = line[0]
            labels.append(label)
            image = line[1:]
            image_int = [int(pixel) for pixel in image]
            image = np.array(image_int)
            images.append(image)

    print("Done!\n")
    return images, labels

#creating k nearest neighbour
def generate_knn_classifier():
    num_neighbours = 10
    print("\nGenerating KNN model with number of neighbours = '{}'...".format(
        num_neighbours))
    classifier_model = KNeighborsClassifier(n_neighbors=num_neighbours)
    print("Done!\n")
    return classifier_model

#creating logistic regresssion classifier
def generate_logistic_classifier():
    print("\nGenerating Logistic-regression model...")
    classifier_model = linear_model.LogisticRegression()
    print("Done!\n")
    return classifier_model

#creating svm
def generate_svm_classifier():
    print("\nGenerating SVM model...")
    classifier_model = svm.LinearSVC()
    print("Done!\n")
    return classifier_model

#create classifier?? not sure how this works since the function name is a string? maybe it just assigns the name of the classifier 
def generate_classifier(model_name):
    classifier_generator_function_name = "generate_{}_classifier".format(
        model_name)
    return globals()[classifier_generator_function_name]()
#globals() The global scope contains all functions, variables which are not associated to any class or function.

#create train and test datasets based on a specified proportion/ratio to include in the train dataset
def divide_data_train_test(images, labels, ratio):
    print("\nDividing dataset in the ratio '{}' using `train_test_split()`:".format(ratio))
    ret = train_test_split(images, labels, test_size=ratio, random_state=0)
    print("Done!\n")
    return ret


def main():
    #sys.argv is a list in Python, which contains the command-line arguments passed to the script. 
    #With the len(sys.argv) function you can count the number of arguments. 
    #If you are gonna work with command line arguments, you probably want to use sys.argv. 
    model_name = sys.argv[1]
    if model_name not in ['svm', 'logistic', 'knn']:
        print("Invalid model-name '{}'!".format(model_name))
        return

    #get model file path
    model_output_dir_path = get_config(
        'model_{}_output_dir_path'.format(model_name))
    model_stats_file_path = os.path.join(
        model_output_dir_path, "stats-{}.txt".format(model_name))
    print("Model stats will be written to the file at path '{}'.".format(
        model_stats_file_path))

    with open(model_stats_file_path, "w") as model_stats_file:
        images_transformed_path = get_config('images_transformed_path')
        images, labels = read_images_transformed(images_transformed_path)
        classifier_model = generate_classifier(model_name) #naming classifier

        model_stats_file.write("Model used = '{}'".format(model_name))
        model_stats_file.write(
            "Classifier model details:\n{}\n\n".format(classifier_model)) #write details into main file
        training_images, testing_images, training_labels, testing_labels = divide_data_train_test(
            images, labels, 0.2) #creating the training dataset

        print("\nTraining the model...")
        classifier_model = classifier_model.fit(
            training_images, training_labels) 
        #it looks like scikit-learn fits the data for almost all the algorithms by name.fit
        print("Done!\n")

        #not sure what dumping the model means....storing? saving new version?
        model_serialized_path = get_config(
            'model_{}_serialized_path'.format(model_name))
        print("\nDumping the trained model to disk at path '{}'...".format(
            model_serialized_path))
        joblib.dump(classifier_model, model_serialized_path) #used instead of pickle
        print("Dumped\n")

        #score(X, y[, sample_weight])	Returns the mean accuracy on the given test data and labels.
        #unsure of the output....is it percentage? magnitude?
        print("\nWriting model stats to file...")
        score = classifier_model.score(testing_images, testing_labels)
        model_stats_file.write(
            "Model score:\n{}\n\n".format(print_with_precision(score)))

        #predict(X)	Perform classification on samples in X.
        predicted = classifier_model.predict(testing_images)
        #metrics.classification_report(y_true, y_pred)	Build a text report showing the main classification metrics
        #Returns:	report : string -->Text summary of the precision, recall, F1 score for each class.
        report = metrics.classification_report(testing_labels, predicted)
        model_stats_file.write(
            "Classification report:\n{}\n\n".format(report))
        print("Done!\n")

        print("\nFinished!\n")

#not sure what this is for
if __name__ == '__main__':
    main()
