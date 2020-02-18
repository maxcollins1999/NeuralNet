from NumNet import NumNet
from ClassicalNet import ClassicalNet
import numpy as np
import phom

np.set_printoptions(threshold=np.inf)

np.random.seed(69)

test = NumNet()

test.loadState()

#print(test.getAccuracy(atype='test'))

#test.go_to_school(.1,1,5000000)

#print(test.getAccuracy(atype='test'))

#est.saveState()

pic = phom.formpho('Num Images\\height.jpg')

phom.showGrayScale(pic)

print(test.classify(pic))

#test.dispState()

#test.go_to_school(.1,10)

#test.dispState()

#test.saveState()

