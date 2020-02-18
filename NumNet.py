### Preamble ###################################################################
#
# Author            : Max Collins
#
# Github            : https://github.com/maxcollins1999
#
# Title             : NumNet2.py 
#
# Date Last Modified: 8/1/2020
#
# Notes             : A basic number classifier without the use of tensorflow
#
################################################################################

### Imports ####################################################################

#Local
import phom
from ClassicalNet import ClassicalNet

#Global
import pathlib
import pickle
import numpy as np
import math
from tqdm import tqdm

################################################################################

### Paths for Pathlib ##########################################################

ptrain_ims = pathlib.Path(__file__).parent / 'Num Images' / 'Train.gz'
ptrain_labels = pathlib.Path(__file__).parent / 'Num Images' / 'Train Labels.gz'
ptest_ims = pathlib.Path(__file__).parent / 'Num Images' / 'Test.gz'
ptest_labels = pathlib.Path(__file__).parent / 'Num Images' / 'Test Labels.gz'

################################################################################

class NumNet(ClassicalNet):

    def __init__(self):

        timages, tlabels  = self.__readPhotos(ptrain_ims,ptrain_labels)
        super().__init__(timages, tlabels,1,16,784,10)

### Public Methods #############################################################    
                
    def getAccuracy(self, data=None, labels=None, atype = None):
        """Extends the super classifier to automatically extract the test data.
        """

        if atype == 'test':
            data,labels = self.__readPhotos(ptest_ims,ptest_labels)
        return super().getAccuracy(data=data, labels=labels)


    def classify(self, data):
        """Extends super classify to return the predicted number.
        """

        data = np.resize(data,(len(data)*len(data[0]),1))
        nums  = super().classify(data)
        number = 0
        conf = 0
        for i, val in enumerate(nums[:,0]):
            if val > conf:
                conf = val
                number = i
        return number

### Private Methods ############################################################
        
    def __readPhotos(self, pims, plabs):
        """Takes in the path of the images and labels and reads in the images 
        that will be used to differentiate numbers and reshapes them into nx1 
        arrays for use in matrix multiplication.
        """
        
        images, labels = phom.getGzipped(pims,plabs)
        for im in images:
            im.resize((len(im)*len(im[0]),1))
        for i, lab in enumerate(labels):
            labels[i] = self.__get_y_mat(lab)
        return images, labels


    def __get_y_mat(self, yval):
        """Takes the y value and returns the y matrix
        """

        y = np.zeros((10,1))
        y[yval,0] = 1
        return y

### Tests ######################################################################
 


################################################################################       