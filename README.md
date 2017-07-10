# SensAI: ASL Translator

**Description:**
</br> Translates American Sign Language (ASL) to text

**Set up Python 2.7 environment in Anaconda:**
</br> *Open Anaconda Prompt*

```
conda create --name python27 python=2.7
activate python27
conda install scikit-learn #includes scipy, numpy (?)
conda install theano
pip install keras
pip install opencv-python
conda install cloudpickle
pip install pygame
pip install pyglet
conda install pillow
pip install matplotlib
conda install lasagne
pip install nolearn
pip install h5py
pip install seaborn
pip install imutils
```

# File Structure

**conf.json** --- *configuration file used to provide inputs to entire system; json file with key-value pair file format to store data effectively*
* model: takes in any of these parameters - inceptionv3, resnet50, vgg16, vgg19 and xception
* weights: takes the value imagenet specifying that we intend to use weights from imagenet; can also set this None if you wish to train the network from scratch
* include_top: takes the value false specifying that we are going to take the features from any intermediate layer of the network; can set this to true if you want to extract features before the fully connected layers
* test_size: takes the value in the range (0.10 - 0.90), to split overall dataset into training and testing 
* seed: takes any value to reproduce same results
* num_classes: specifies the number of classes considered for the image classification problem

# Tracker

**Key Annotations In Code:**
* EDIT! ~ edit
* [?????] or ? ~ question

**Commits after training:**
* *Scenario 1*
</br> Dataset: signs1 (ASL3)
</br> Pre-trained model: vgg16
</br> Feature extraction time: 17:26-18:34
</br> Models/results from test sizes: 0.1, 0.3, 0.5, 0.7, 0.9, 0.95
* *Scenario 2*
</br> Dataset: signs1&2 (ASL3, ASL1)
</br> Pre-trained model: vgg16
</br> Feature extraction time: ---
</br> Models/results from test sizes: 0.1, 0.3, 0.5, 0.7, 0.9, 0.95

# Thanks to the following:

**Mentors, Sponsors, Support:**
* [McGill Innovation Team & Participants](https://www.mcgill-innovation.com/ai-summer-lab)
* [McGill Reasoning & Learning Lab](http://rl.cs.mcgill.ca/)
* [CIFAR](https://www.cifar.ca/)
* [Ethan Macdonald](https://www.linkedin.com/in/ethanbrycemacdonald/)
* [Negar Rostamzadeh](https://www.linkedin.com/in/nrostamzadeh/)
* [Pedro Oliveira Pinheiro](https://www.linkedin.com/in/pedro-oliveira-pinheiro-54630229/)
* Anmol Jawandha
* Perouz Taslakian
* [Alexis Smirnov](https://www.linkedin.com/in/alexissmirnov/)
* [Nicolas Le Roux](https://www.linkedin.com/in/lerouxni/)
* [Gheorghe Comanici](https://www.linkedin.com/in/gheorghe-comanici-b26819103/)
* [Adriana Romero Soriano](https://www.linkedin.com/in/adriana-romero-a6415123/)

**Resources:**
* [Flower Recognition Deep Learning](https://gogul09.github.io/software/flower-recognition-deep-learning)