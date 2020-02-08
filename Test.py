from NumNet2 import NumNet2
import numpy as np



test = NumNet2(1,16,784,10)

x = test.forwardProp(test.train_images[0])

#test.dispState()

y = np.zeros((10,1))

y[test.train_labels[0]] = 1


a,b = test.backProp(x,y)

np.set_printoptions(threshold=np.inf)

for mat in a:
    print(mat)
    print(mat.shape)

#test.dispState()

#test.saveState()

