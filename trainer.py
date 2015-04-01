# -*- coding: utf-8 -*-
"""
Training File for the Neural Network
Created by Duncan Campbell

This file is used to train the neural network for the project
"""

from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, \
                              FullConnection
import cPickle
import os.path as path

# Create the neural network and the layers
network = FeedForwardNetwork()
inLayer = LinearLayer(1024)
hiddenLayer1 = SigmoidLayer(1024*2)
hiddenLayer2 = SigmoidLayer(1024*3)
hiddenLayer3 = SigmoidLayer(1024*2)
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

# Load in CIFAR10 dataset using cPcikle
dataFile1 = open(path.join('cifar-10-batches-py', '))