import numpy as np
import pandas as pd
from random import randint
#

class RTLearner(object):

    def __init__(self, leaf_size, verbose=False, tree=None):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = tree

        if verbose:
            print ("leaf_size is :  ", self.leaf_size)
            print (" Building tree not a forest...... \n")
            #print np.shape(self.tree)
        else:
            pass

    def author(self):
        return "zwin3"

    def get_indx(self, x_train, numberOfSamples):
        leaf_indx =  np.random.randint(x_train.shape[1], size=1) # random indx for splitting randint(0, x_train.shape[1] -1 )
        split_val = np.mean(x_train[:,leaf_indx])
        return leaf_indx[0]

    def make_DT_not_forest(self, Dataset ):
        x_train = Dataset[:, 0:-1]
        y_train = Dataset[:, -1:]
        numberOfSamples = x_train.shape[0]
        number_features = x_train.shape[1]
        length =  np.random.randint(x_train.shape[1])
        if ( len(y_train) == 1 ):   # labels are the same -> Base recursion
            split_val = np.mean(y_train[:,])
            return np.vstack(( self.tree ,  np.array( [['leaf' , split_val,  np.nan,  np.nan]] )))
        elif  (x_train.shape[0] <= self.leaf_size) :   # Base recursion
            split_val = np.mean(y_train[:,])
            return np.vstack(( self.tree , np.array(  [['leaf', split_val, np.nan, np.nan]] )))
        else:
            split_val = np.median(Dataset[:, length])
            right_tree_shape = ( Dataset[Dataset[:, length] > split_val]).shape[0]
            if (right_tree_shape == 0):   # The end of tree
                split_val = np.mean(Dataset[:, length])
                leaf_val = y_train.mean()
                return np.vstack((self.tree, np.array([['leaf', leaf_val, np.nan, np.nan]])))
            left = Dataset[:, length] <= split_val
            right = ~left  # complement of left
            leftTree = self.make_DT_not_forest(Dataset[left])
            rightTree = self.make_DT_not_forest(Dataset[right])
            root = np.array([[length, split_val, 1, leftTree.shape[0] + 1]])
            return np.vstack((self.tree, root, leftTree, rightTree))

    def addEvidence(self, Xtrain, Ytrain):
        self.tree = np.array([]).reshape((0,4))
        Dataset = np.column_stack((Xtrain, Ytrain))
        #Building the model using recursive calls
        self.tree = self.make_DT_not_forest(Dataset)
        if self.verbose:
            print ("Tree shape is :  ", np.shape(self.tree))

    def query(self, Xtest):
        result = np.empty( [Xtest.shape[0], 1] , dtype = float)
        for indx, row in enumerate(Xtest):
            init_index = 0
            temp_leaf = self.tree[ init_index , 0]
            while ( temp_leaf != 'leaf'):
                if (  row[int(float(temp_leaf) ) ] <= float( self.tree[init_index , 1] ) ):   # left side add indx + 1
                    init_index += 1
                else:                   # right side add indx + left side
                    init_index += int( float( self.tree[init_index , 3] ) )
                temp_leaf = self.tree[init_index , 0]
            result[indx, 0] = self.tree[init_index , 1]
        return result.flatten()

if __name__=="__main__":
    print "the secret clue is RTLearner\n"
