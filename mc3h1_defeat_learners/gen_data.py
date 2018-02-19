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
    row_number = np.random.randint(20,1001)
    col_number = np.random.randint(2,1001)
    Y = np.zeros(row_number)
    X = np.random.normal(size = (row_number, col_number))
    for i in range(0,row_number):
        Y = Y + X[:,i]
    return X, Y

def best4DT(seed=1489683273):
    np.random.seed(seed)
    #X = np.zeros((100,2))
    #Y = np.random.random(size = (100,))*200-100
    X = np.random.normal(size = (np.random.randint(20,1001),np.random.randint(2,1001) ))
    Y = np.zeros(np.random.randint(20,1001))
    for j in range(0,np.random.randint(2,1001)):
        Y = Y + X[:,j] **2
    return X, Y

def author():
    return 'zwin3' #Change this to your user ID

if __name__=="__main__":
    print "they call me Bro."
