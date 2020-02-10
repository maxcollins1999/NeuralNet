from NumNet2 import NumNet2
import numpy as np

np.set_printoptions(threshold=np.inf)

np.random.seed(69)

test = NumNet2(1,16,784,10)

#x = test.forwardProp(test.train_images[0])

#test.dispState()

#y = np.zeros((10,1))

#y[test.train_labels[0]] = 1


#a,b = test.backProp(x,y)

print(test.getAccuracy())

#print(test.train_images[0])

#print(test.wmats[0]@test.train_images[0])

amats = test.forwardProp(test.train_images[0])

test.go_to_school(.5, 100)

print(test.getAccuracy())

#test.dispState()

#test.go_to_school(.1,10)

#test.dispState()

#test.saveState()

