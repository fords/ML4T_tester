"""
template for generating data to fool learners (c) 2016 Tucker Balch
"""

import numpy as np
import math

# this function should return a dataset (X and Y) that will work
# better for linear regression than decision trees
def best4LinReg(seed=1489683273):
    np.random.seed(seed)
    #X = np.zeros((100,2))
    #Y = np.random.random(size = (100,))*200-100
    # Here's is an example of creating a Y from randomly generated
    # X with multiple columns
    # Y = X[:,0] + np.sin(X[:,1]) + X[:,2]**2 + X[:,3]**3
    X = np.random.random(size = (100,50))*200-100
    Y = np.zeros((100,))
    C = np.random.random(size = (50,)) * 100
    for i in range(0, 49):
        Y += C[i] * X[:, i]
    return X, Y

def best4DT(seed=1489683273):
    np.random.seed(seed)
    #X = np.zeros((100,2))
    #Y = np.random.random(size = (100,))*200-100
    np.random.seed(seed)
    columns = 100
    rows = 100
    X = np.random.random(size = (rows,columns))*200-100
    Y = np.zeros((rows,))
    for i in range(0, (columns - 1)):
        if (i%5 == 0):
            Y += np.sin(X[:, i])
        else:
            Y += np.power(X[:, i], ((1/7)))
    return X, Y


def author():
    return 'zwin3' #Change this to your user ID

if __name__=="__main__":
    print "they call me Bro."
