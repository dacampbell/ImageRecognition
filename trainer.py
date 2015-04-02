# -*- coding: utf-8 -*-
"""
Training File for the Neural Network
Created by Duncan Campbell

This file is used to train the neural network for the project
"""

from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, \
                              FullConnection
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer


import cPickle
import os.path as path
import numpy as np

# Create the neural network and the layers
print 'Initializing the Neural Network'
network = FeedForwardNetwork()
inLayer = LinearLayer(3072)
hiddenLayer1 = SigmoidLayer(1000)
hiddenLayer2 = SigmoidLayer(500)
hiddenLayer3 = SigmoidLayer(250)
outLayer = LinearLayer(10)

# Add the layers to the neural network
network.addInputModule(inLayer)
network.addOutputModule(outLayer)
network.addModule(hiddenLayer1)
network.addModule(hiddenLayer2)
network.addModule(hiddenLayer3)

# Create and add the connections to the neural network
in_to_hidden1 = FullConnection(inLayer, hiddenLayer1)
hidden1_to_hidden2 = FullConnection(hiddenLayer1, hiddenLayer2)
hidden2_to_hidden3 = FullConnection(hiddenLayer2, hiddenLayer3)
hidden3_to_out = FullConnection(hiddenLayer3, outLayer)

network.addConnection(in_to_hidden1)
network.addConnection(hidden1_to_hidden2)
network.addConnection(hidden2_to_hidden3)
network.addConnection(hidden3_to_out)

# Neural network performs internal initialization
network.sortModules()

# Load in CIFAR10 dataset using cPcikle and conver to NumPy array
dataFile1 = open(path.join('cifar-10-batches-py', 'data_batch_1'), 'rd')
data = cPickle.load(dataFile1)
images = np.array(data['data'])
labels = np.array(data['labels'])

# Construct the supervised data set for learning
print 'Constructing the Data Set'
dataSet = ClassificationDataSet(3072, 1, nb_classes = 10)

for index in range(0, labels.size):
    dataSet.addSample(images[index], labels[index])
dataSet._convertToOneOfMany();
    

# Train the neural network
print 'Training the Neural Network'
trainer = BackpropTrainer(network, dataset = dataSet, momentum = 0.1, verbose = True, weightdecay = 0.01)
trainer.trainEpochs(5)

# Save the neural network to a file for later use
print 'Saving to File'
networkFile = open('trainedNet1.cpkl', 'w')
cPickle.dump(network, networkFile)

print 'Finished Training Network'
