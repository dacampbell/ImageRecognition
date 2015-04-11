# -*- coding: utf-8 -*-
"""
Evaluate the success of the neural network
Created by Duncan Campbell

This file is used to evaluate the generated neural network on test data
"""

from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer

import cPickle
import numpy as np
import os.path as path

# Loads the neural network from a file
print "Loading Network File"
networkFile = open('trainedNet1.cpkl', 'rd')
network = cPickle.load(networkFile)

# Load in the data
dataFile1 = open(path.join('cifar-10-batches-py', 'data_batch_2'), 'rd')
data = cPickle.load(dataFile1)
images = np.array(data['data'])
labels = np.array(data['labels'])

# Construct the classification data set for evaluation
print 'Constructing the Data Set'
dataSet = ClassificationDataSet(3072, 1, nb_classes = 10)

for index in range(0, labels.size):
    dataSet.addSample(images[index], labels[index])
dataSet._convertToOneOfMany();

# Create the trainer to use to evaluate the existing network
print 'Creating the Trainer'
trainer = BackpropTrainer(network)

# Evaluate the network against the test data
print 'Testing the data'
percentError = 100 * trainer.testOnData(dataset = dataSet)
print 'Data had', percentError, 'error'

