### Preamble ###################################################################
#
# Author            : Max Collins
#
# Github            : https://github.com/maxcollins1999
#
# Title             : ClassicalNet.py 
#
# Date Last Modified: 8/1/2020
#
# Notes             : A basic neural network classifier without the use of 
#                     tensorflow
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

################################################################################

class ClassicalNet:
    
    def __init__(self, tdata, tlabels, noh=None, nih=None, nin=None, nou=None):
        """Takes the number of hidden networks (noh), the number of nodes in 
        hidden network (nih), number in (nin) and number out (nou), a list of 2D 
        numpy arrays train data and list of 2D numpy arrays train labels and 
        constructs the initial b matrices.
        """

        self.wmats = [] #Edge weights
        self.bmats = [] #b matrices 

        self.train_data = tdata
        self.train_labels = tlabels 

        if noh and nih and nin and nou:
            self.wmats.append(np.random.randn(nih,nin))
            self.bmats.append(np.random.randn(nih,1))
            for i in range(0,noh-1):
                self.wmats.append(np.random.randn(nih,nih))
                self.bmats.append(np.random.randn(nih,1))
            self.wmats.append(np.random.randn(nou,nih))
            self.bmats.append(np.random.randn(nou,1))
        elif not (not noh and not nih and not nin and not nou):
            raise AttributeError('Must assign complete network scaffold or no'+\
                                 ' scaffold')


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
            cost_deriv.insert(0, delta_list[0]@amats[i].T)
        return delta_list, cost_deriv 


    def go_to_school(self, alpha, b_size, iter):
        """Takes the gradient descent stepsize and the batch size and the 
        number of iterations and performs back and forward propogation. 
        """

        im_batch, lab_batch = self.__calc_batch(b_size)
        i = 0
        for t in tqdm(range(0, iter)):
            if i>=len(im_batch):
                i = 0
            y= lab_batch[i][0]
            amats = self.forwardProp(im_batch[i][0])
            delta, cost = self.backProp(amats,y)
            for k, im in enumerate(im_batch[i][1:]):
                y = lab_batch[k][0]
                amats = self.forwardProp(im)
                delta_list, cost_deriv = self.backProp(amats,y)
                for j, mat in enumerate(delta_list):
                    delta[j] = delta[j] + mat
                for j, mat in enumerate(cost_deriv):
                    cost[j] = cost[j] + mat
            for k, mat in enumerate(delta):
                delta[k] = mat / len(im_batch[i])
                self.bmats[k] = self.bmats[k] - alpha*delta[k]
            for k, mat in enumerate(cost):
                cost[k] = mat / len(im_batch[i])
                self.wmats[k] = self.wmats[k] - alpha*cost[k]
            i+=1


    def getAccuracy(self, data = None, labels = None):
        """Determines the current program accuracy
        """

        if data is None or labels is None:
            print('here')
            data = self.train_data
            labels = self.train_labels
        count = 0
        for i, im in enumerate(tqdm(data)):
            amats = self.forwardProp(im)
            number1 = None
            number2 = None
            conf1 = 0
            conf2 = 0
            for k, val in enumerate(amats[-1][:,0]):
                if val > conf1:
                    number1 = k
                    conf1 = val
                if labels[i][k,0] > conf2:
                    number2 = k
                    conf2 = labels[i][k,0]
            if number1 == number2:
                count+=1
        return count/len(data)


    def classify(self, data):
        """Takes a nx1 numpy array of data and returns the neural net 
        classification.
        """

        amats = self.forwardProp(data)
        return amats[-1]
                

### Private Methods ############################################################

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

        im_batch = [self.train_data[i * n:(i + 1) * n] for i in range((len(self.train_data) + n - 1) // n )]  
        lab_batch = [self.train_labels[i * n:(i + 1) * n] for i in range((len(self.train_data) + n - 1) // n )]  
        return im_batch, lab_batch

### Tests ######################################################################
 


################################################################################       