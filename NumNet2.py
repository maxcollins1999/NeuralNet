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
    train_images = [] #List of training images
    train_labels = [] #List of corresponding training labels
    
    def __init__(self, noh=None, nih=None, nin=None, nou=None):
        """Takes the number of hidden networks (noh), the number of nodes in 
        hidden network (nih), number in (nin) and number out (nou), and 
        constructs the initial b matrices.
        """

        if noh and nih and nin and nou:
            self.wmats.append(np.random.rand(nih,nin))
            self.bmats.append(np.random.rand(nih,1))
            for i in range(0,noh-1):
                self.wmats.append(np.random.rand(nih,nih))
                self.bmats.append(np.random.rand(nih,1))
            self.wmats.append(np.random.rand(nou,nih))
            self.bmats.append(np.random.rand(nou,1))
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


    def forwardProp(self, a1mat):
        """Takes the initial input and performs forward propagation and returns 
        a list of a matrices
        """

        amats = []
        amats.append(a1mat)
        for i, wmat in enumerate(self.wmats):
            amats.append(self.__sigmoid(wmat@amats[i]+self.bmats[i]))
        return amats


    def backProp(self, amats, ymat):
        """Takes a list of amatrices and a nx1 ymatrix and performs an 
        iteration of backpropogation.
        """

        delta_list = [(amats[-1]-ymat)*self.__sigmoid_prime(self.wmats[-1]@amats[-2]+self.bmats[-1])]
        cost_deriv = [delta_list[0]@amats[-2].T]
        for i, wmat in reversed(list(enumerate(self.wmats[:-1]))):
            delta_list.insert(0, (self.wmats[i+1].T@delta_list[0])*self.__sigmoid_prime(wmat@amats[i]+self.bmats[i]))

            print(wmat@amats[i]+self.bmats[i])

            cost_deriv.insert(0, delta_list[0]@amats[i].T)
        return delta_list, cost_deriv 


    def go_to_school(self, alpha, iter):
        """Takes the gradient descent stepsize and the number of iterations and 
        performs back and forward propogation. 
        """

        i = 0
        for t in tqdm(range(0, iter)):
            if i>=len(self.train_images):
                i = 0
            y= self.__get_y_mat(self.train_labels[i])
            amats = self.forwardProp(self.train_images[i])
            delta, cost = self.backProp(amats,y)
            for k, mat in enumerate(delta):
                self.bmats[k] = self.bmats[k] - alpha*delta[k]
            for k, mat in enumerate(cost):
                self.wmats[k] = self.wmats[k] - alpha*cost[k]
            i+=1


    def getAccuracy(self):
        """Determines the current program accuracy
        """

        count = 0
        for i, im in enumerate(tqdm(self.train_images)):
            amats = self.forwardProp(im)
            number = None
            conf = 0
            for k, val in enumerate(amats[-1][:,0]):
                if val > conf:
                    number = k
                    conf = val
            if number == self.train_labels[i]:
                count+=1
        return count/len(self.train_images)
                

### Private Methods ############################################################
        
    def __readPhotos(self):
        """Takes in the number of photos to be read and reads in the images 
        that will be used to differentiate numbers and reshapes them into nx1 
        arrays for use in matrix multiplication.
        """
        
        self.train_images, self.train_labels = phom.getGzipped(train_ims,train_labels)
        for im in self.train_images:
            im.resize((len(im)*len(im[0]),1))


    def __sigmoid(self,val):
        """Returns the value of val on the sigmoid curve
        """
        
        return 1/(1+np.exp(-val))


    def __sigmoid_prime(self,val):
        """Returns the value of the derivative of the sigmoid function
        """
        return self.__sigmoid(val)*(1-self.__sigmoid(val))


    def __calc_batch(self, n):
        """Takes the size of each batch and randomly splits the training set 
        into batches that are of the correct size and returns the results.
        """

        im_batch = [self.train_images[i * n:(i + 1) * n] for i in range((len(self.train_images) + n - 1) // n )]  
        lab_batch = [self.train_labels[i * n:(i + 1) * n] for i in range((len(self.train_labels) + n - 1) // n )]  
        return im_batch, lab_batch


    def __get_y_mat(self, yval):
        """Takes the y value and returns the y matrix
        """

        nums = self.bmats[-1].shape[0]
        y = np.zeros((nums,1))
        y[yval,0] = 1
        return y

### Tests ######################################################################
 


################################################################################       