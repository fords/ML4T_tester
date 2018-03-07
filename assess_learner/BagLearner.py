import numpy as np
import pandas as pd
import LinRegLearner as lrl
import DTLearner as dt
class BagLearner(object):

    def __init__(self, learner, kwargs , bags = 20 , boost = False, verbose = False ):
        self.learner = learner
        self.bags = bags
        temp_learner = []
        for i in range(self.bags):
            temp_learner.append(learner(**kwargs))
        self.learner = temp_learner
        self.kwargs = kwargs
        self.boost = boost
        self.verbose = verbose
        self.Xbags = None
        self.Ybags = None
        if verbose:
            print 'Printing Debugging info....'

    def author():
            return 'zwin3'

    def addEvidence(self,Xtrain, Ytrain):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        numberOfSamples = self.Xtrain.shape[0]
        index = []
        self.Xbags = []
        self.Ybags = []

        for learner in self.learner:
            idx = np.random.choice(numberOfSamples, numberOfSamples)
            self.Xbags = Xtrain[idx]
            self.Ybags = Ytrain[idx]
            learner.addEvidence(self.Xbags, self.Ybags)


    def query(self,Xtest):
        learner  = []
        self.Xtest = Xtest
        Arra = np.array([learner.query(Xtest) for learner in self.learner])
        return np.mean(Arra, axis=0)

        """result = np.zeros((len(self.Xbags), len(Xtest)))
        for i in range(0, self.bags):
            learner.append(self.learner(self.kwargs))

        for i in range(0,len(learners)):
            learner[i].addEvidence(self.Xbags[i],self.Ybags[i])

            result[i] = learner[i].query(self.Xtest)
        return np.mean(result, axis = 0)"""

if __name__=="__main__":
    print ("secret clue is a BagLearner\n")
