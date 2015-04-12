# -*- coding: utf-8 -*-
"""
Automated Classifier
Shows the gui which runs a Demo to run the classifier code live
Created by Duncan Campbell
"""

import sys
import numpy as np
import cPickle
import os.path as path

from PyQt4 import QtGui, QtCore
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError

class MainWindow(QtGui.QWidget):    
    
    def __init__(self):
        super(MainWindow, self).__init__()
        self.init()
        
    def init(self):
        self.classifyButton = QtGui.QPushButton('Classify', self)
        self.classifyButton.setGeometry(10, 10, 100, 25)
        self.connect(self.classifyButton, QtCore.SIGNAL('clicked()'), self.classifyImage)
        
        self.nextImageButton = QtGui.QPushButton('Next Image', self)
        self.nextImageButton.setGeometry(125, 10, 100, 25)
        self.connect(self.nextImageButton, QtCore.SIGNAL('clicked()'), self.nextImage)
        
        self.classificationText = QtGui.QLabel(self)
        self.classificationText.setGeometry(250, 10, 200, 25)
        
        self.displayImage = QtGui.QLabel(self)
        self.displayImage.move(10, 50)
        
        self.setGeometry(300, 300, 500, 500)
        self.setWindowTitle('Image Classifier')
        self.show()
        
        self.loadImages()
        self.loadNetwork()
        
        self.currImage = 0
    
    def loadNetwork(self):
        networkFile = open('trainedNet1.cpkl', 'r')
        network = cPickle.load(networkFile)
        networkFile.close()
        
        self.trainer = BackpropTrainer(network)
                
    def loadImages(self):
        dataFile = open(path.join('cifar-10-batches-py', 'test_batch'), 'r')
        data = cPickle.load(dataFile)
        self.images = np.array(data['data'])
        self.labels = np.array(data['labels'])
        dataFile.close()
        
    def drawImage(self):
        imageData = self.images[self.currImage]        
        self.currImage += 1
        if(self.currImage >= self.labels.size):
            self.currImage = 0;
            
        image = QtGui.QImage(32, 32, QtGui.QImage.Format_RGB32)
        
        for r in range(0, 32):
            for c in range(0, 32):
                color = QtGui.qRgb(imageData[r + c * 32], imageData[r + c * 32 + 1024], imageData[r + c * 32 + 2048])
                image.setPixel(r, c, color)
        image = image.scaledToHeight(400)
        
        
        self.displayImage.setPixmap(QtGui.QPixmap.fromImage(image))
        self.displayImage.resize(400, 400)
        self.displayImage.show()
    
    def classifyUsingNetwork(self):
        dataSet = ClassificationDataSet(3072, 1, nb_classes = 10)
        dataSet.addSample(self.images[self.currImage], self.labels[self.currImage])
        dataSet._convertToOneOfMany();
        
        index = percentError( self.trainer.testOnClassData(
           dataset=dataSet ), dataSet['class'] )
        
        print index
        
        return index
        
        
    def classifyImage(self):
        num = self.classifyUsingNetwork()
        
        
        if(num == 0.0):
            self.classificationText.setText('Correctly Idenified')
        else:
            self.classificationText.setText('Not Correctly Idenified')
            
        
    def nextImage(self):
        self.classificationText.setText('')
        self.drawImage()
        

application = QtGui.QApplication(sys.argv)
mainWindow = MainWindow()
sys.exit(application.exec_())