import numpy as np
from NumNet2 import NumNet2


def __sigmoid(val):
    """Returns the value of val on the sigmoid curve
    """

    return 1/(1+np.exp(-val))


def __sigmoid_prime(val):
    """Returns the value of the derivative of the sigmoid function
    """
    return __sigmoid(val)*(1-__sigmoid(val))

a0 = np.array([[.05],[.1]])

w0 = np.array([[0.15,0.20],[0.25,0.30]])

b0 = np.array([[.35],[.35]])

a1 = __sigmoid(w0@a0 + b0)

w1 = np.array([[0.40,0.45],[0.50,0.55]])

b1 = np.array([[0.6],[0.60]])

y = np.array([[.01],[.99]])

a2 = __sigmoid(w1@a1+b1)

delta1 = (a2 - y)*__sigmoid_prime(w1@a1+b1)

delta0 = w1.T@delta1*__sigmoid_prime(w0@a0+b0)

#print(w0@a0 + b0)

#print(delta0)

#print(delta1)

#print(delta0@a0.T)

#print(delta1@a1.T)

#################

amats = [a0,a1,a2]

np.random.seed(69)

test = NumNet2(1,2,2,2)

test.wmats = [w0,w1]
test.bmats = [b0,b1]

#test.dispState()

delta, cost = test.backProp(amats,y)

print(delta[0])

print(cost[0])

