# SensAI: ASL Translator

**Description:**
</br> Translates American Sign Language (ASL) to text

**Key Annotations In Code:**
</br> EDIT! ~ edit
</br> [?????] or ? ~ question

**Set up Python 3.5 environment in Anaconda:**
</br> *Open Anaconda Prompt*
</br> conda create --name python35 python=3.5
</br> activate python35
</br> conda install scikit-learn #includes scipy, numpy (?)
</br> conda install theano
</br> conda install -c conda-forge tensorflow **OR** pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.2.0-cp35-cp35m-win_amd64.whl
</br> pip install keras
</br> pip install opencv-python
</br> pip install pyyaml matplotlib
</br> conda install cloudpickle

**Set up Python 2.7 environment in Anaconda:**
</br> *Open Anaconda Prompt*
</br> conda create --name python27 python=2.7
</br> activate python27
</br> conda install scikit-learn #includes scipy, numpy (?)
</br> conda install theano
</br> pip install keras
</br> pip install opencv-python
</br> conda install cloudpickle
</br> pip install pygame
</br> pip install pyglet
</br> conda install pillow
</br> pip install matplotlib
</br> conda install lasagne
</br> pip install nolearn
</br> pip install h5py
</br> pip install seaborn
</br> pip install imutils

**To add to environment:**
</br> conda install jupyter
</br> conda install spyder
</br> pip install argparse

**File structure:**
</br> *conf.json* --- configuration file used to provide inputs to entire system; json file with key-value pair file format to store data effectively
</br> The model key takes in any of these parameters - inceptionv3, resnet50, vgg16, vgg19 and xception.
</br> The weights key takes the value imagenet specifying that we intend to use weights from imagenet. You can also set this None if you wish to train the network from scratch.
</br> The include_top key takes the value false specifying that we are going to take the features from any intermediate layer of the network. You can set this to true if you want to extract features before the fully connected layers.
</br> The test_size key takes the value in the range (0.10 - 0.90). This is to make a split between your overall data into training and testing.
</br> The seed key takes any value to reproduce same results.
</br> The num_classes specifies the number of classes considered for the image classification problem.

**Commits:**
</br> *After training:*
</br> Dataset: signs1 (ASL3)
</br> Pre-trained model: vgg16
</br> Feature extraction time: 17:26-18:34
</br> Models/results from test sizes: 0.1, 0.3, 0.5, 0.7, 0.9, 0.95

**Resoruces:**
</br> https://gogul09.github.io/software/flower-recognition-deep-learning

**Thoughts:**
</br> *Segment the Hand region:*
</br> 1. Background Subtraction
</br> 2. Motion Detection and Thresholding
</br> 3. Contour Extraction
</br> 
</br> 