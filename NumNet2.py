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

#Global
import pathlib
import pickle
import numpy as np
import math
from tqdm import tqdm
import copy

################################################################################

### Paths for Pathlib ##########################################################

train_ims = pathlib.Path(__file__).parent / 'Num Images' / 'Train.gz'
train_labels = pathlib.Path(__file__).parent / 'Num Images' / 'Train Labels.gz'
test_ims = pathlib.Path(__file__).parent / 'Num Images' / 'Test.gz'
test_labels = pathlib.Path(__file__).parent / 'Num Images' / 'Test Labels.gz'

################################################################################

class NumNet2:
    
    wmats = []    #List containing the numpy arrays of the edge weights
    bmats = []    #List containign the numpy arrays of the b matrices
    sig_param = 1 #Sigmoid function k parameter
    train_images = [] #List of training images
    train_labels = [] #List of corresponding training labels
    
    def __init__(self, noh=None, nih=None, nin=None, nou=None):
        """Takes the number of hidden networks (noh), the number of nodes in 
        hidden network (nih), number in (nin) and number out (nou), and 
        constructs the initial b matrices.
        """

        
        if noh and nih and nin and nou:
            self.wmats.append(np.zeros((nih,nin)))
            self.bmats.append(np.zeros((nih,1)))
            for i in range(0,noh-1):
                self.wmats.append(np.zeros((nih,nih)))
                self.bmats.append(np.zeros((nih,1)))
            self.wmats.append(np.zeros((nou,nih)))
            self.bmats.append(np.zeros((nou,1)))
        elif not (not noh and not nih and not nin and not nou):
            raise AttributeError('Must assign complete network scaffold or no'+\
                                 ' scaffold')
        self.__readPhotos()


### Public Methods #############################################################    
    
    def dispState(self):
        """Displays the current values for each of the b matrices
        """
        for i, w in enumerate(self.wmats):
            print('-----------------------')
            print('w'+str(i)+'\n')
            print(w)
            print(w.shape)
            print('-----------------------')
            print('b'+str(i)+'\n')
            print(self.bmats[i])
            print(self.bmats[i].shape)


    def saveState(self):
        """Saves the current object state as a pickle binary file
        """

        dump = {
            'wmats':self.wmats,
            'bmats':self.bmats
        }
        with open('save.txt','wb') as fstrm:
            pickle.dump(dump,fstrm)


    def loadState(self):
        """Loads the current object state from the pickle binary file
        """

        with open('save.txt','rb') as fstrm:
            dump = pickle.load(fstrm)
        self.wmats = dump['wmats']
        self.bmats = dump['bmats']

### Private Methods ############################################################
        
    def __readPhotos(self):
        """Takes in the number of photos to be read and reads in the images 
        that will be used to differentiate numbers and reshapes them into nx1 
        arrays for use in matrix multiplication.
        """
        
        self.train_images, self.train_labels = phom.getGzipped(train_ims,train_labels)
        for im in self.train_images:
            im.resize((len(im)*len(im[0]),1))

### Tests ######################################################################
 


################################################################################       