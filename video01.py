#cd C:/Users/otterrafting/Dropbox/AA McGill AI/ASL
#cd C:/Users/otterrafting/Documents/GitHub/SensAI
#import numpy as np #not needed
import cv2

#import os
#import sys
##The traceback module works with the call stack to produce error messages. 
#import traceback

##import pickle
#import cloudpickle as pickle
#import cv2
#from sklearn.externals import joblib
##import joblib

##having difficulty figuring out what the common package does
#from common.config import get_config
#from common.image_transformation import apply_image_transformation
#from common.image_transformation import resize_image


cap = cv2.VideoCapture(	1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here - cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)

    #check with model!!! from ASL1

    #try:
    #        frame = apply_image_transformation(frame)
    #        frame_flattened = frame.flatten() #numpy - collapses array into one dimension
    #        #print(model_serialized_path)
    #        #model_serialized_path = "C:\\Users\\AanikaRahman\\Documents\\GitHub\\ASL1\\Sign-Language-Recognition\\data\\generated\\output\\svm\\model-serialized-svm.pkl"
    #        classifier_model = joblib.load(model_serialized_path)
            #classifier_model = pickle.load(open(model_serialized_path),encoding='latin1')
    #        predicted_labels = classifier_model.predict(frame_flattened)
    #        predicted_label = predicted_labels[0]
    #        print("Predicted label = {}".format(predicted_label))
    #        predicted_image = get_image_from_label(predicted_label) #unsure why predicted image has its value set twice. i guess images have multiple elements to them?
    #        predicted_image = resize_image(predicted_image, 200)
    #        cv2.imshow("Prediction = '{}'".format(predicted_label), predicted_image)
    #   except Exception:
    #        exception_traceback = traceback.format_exc()
    #        print("Error while applying image transformation with the following exception trace:\n{}".format(
    #            exception_traceback))

    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

    #save frame
    cv2.imwrite('frame.png',frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()