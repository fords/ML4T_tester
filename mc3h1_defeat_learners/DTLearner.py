"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch
Note, this is NOT a correct DTLearner; Replace with your own implementation.
"""

import numpy as np
import warnings
import math

class DTLearner(object):

    def __init__(self, leaf_size=1, verbose = False):
        warnings.warn("\n\n  WARNING! THIS IS NOT A CORRECT DTLearner IMPLEMENTATION! REPLACE WITH YOUR OWN CODE\n")
        pass # move along, these aren't the drones you're looking for

    def author(self):
        return 'zwin3' # replace tb34 with your Georgia Tech username

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.model = np.array([]).reshape((0,4))
        data = np.column_stack((dataX, dataY))
        self.model = np.concatenate([self.model, self.createDT(data)], axis=0)
        # slap on 1s column so linear regression finds a constant term
        #newdataX = np.ones([dataX.shape[0],dataX.shape[1]+1])
        #newdataX[:,0:dataX.shape[1]]=dataX

        # build and save the model
        #self.model_coefs, residuals, rank, s = np.linalg.lstsq(newdataX, dataY)
    def createDT(self, data):
        x = data[:, 0:-1]
        y = data[:, -1:]
        if (x.shape[0] <= self.leaf_size) or (np.all(y == y[0])):
            split_val = y.mean()
            return np.vstack((self.model, np.array([['leaf', split_val, np.nan, np.nan]])))
        else:
            max_index = self.get_index(x, y)
            split_val = np.median(data[:, max_index])
            lefttree_shape = (data[data[:, max_index] <= split_val]).shape[0]
            righttree_shape = (data[data[:, max_index] > split_val]).shape[0]
            if (lefttree_shape == data.shape[0]) or (righttree_shape == data.shape[0]):
                split_val = np.mean(data[:, max_index])
                leaf_val = y.mean()
                return np.vstack((self.model, np.array([['leaf', leaf_val, np.nan, np.nan]])))
            lefttree = self.createDT(data[data[:, max_index] <= split_val])
            righttree = self.createDT(data[data[:, max_index] > split_val])
            root = np.array([[max_index, split_val, 1, lefttree.shape[0] + 1]])
            return np.vstack((self.model, root, lefttree, righttree))

    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        decision = np.empty([points.shape[0], 1], dtype = float)
        for index, row in enumerate(points):
            start_index = 0
            factor = self.model[start_index, 0]
            while (factor != 'leaf'):
                factor_int = int(float(factor))
                if (row[factor_int] <= float(self.model[start_index, 1])):
                    start_index += 1
                else:
                    start_index += int(float(self.model[start_index, 3]))
                factor = self.model[start_index, 0]

            decision[index, 0] = float(self.model[start_index, 1])

        return decision.flatten()

    def get_index(self, x, y):
        corrs = abs(np.array([self.get_correlation(x_attr, y.T) for x_attr in x.T]))
        corr_matrix = np.zeros((x.shape[1], 0))
        for index, corr in enumerate(corrs):
            if(not(math.isnan(corr[0, 1]))):
                corr_matrix = np.append(corr_matrix, np.array(corr[0,1]))
            else:
                corr_matrix = np.append(corr_matrix, np.array([0]))
        selected_attr = np.argmax(corr_matrix)
        return selected_attr

    def get_correlation(self, x, y):
        return np.corrcoef(x, y)

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
