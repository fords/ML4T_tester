import numpy as np
import pandas as pd
import LinRegLearner as lrl
import DTLearner as dt
import BagLearner as bl
class InsaneLearner(object):
    def __init__(self, verbose):
        self.verbose= verbose
        self.learner = []
        if self.verbose:
            print ' Debugging .....  '
        pass

    def author(self):
        return 'zwin3'

    def addEvidence(self, x, y):
        for i in range(0,20):  # bags are fixed 20 
            learner = bl.BagLearner( learner = lrl.LinRegLearner, kwargs = {}, bags = 20, boost = False, verbose = False)
            learner.addEvidence(x, y)
            self.learner.append(learner)

    def query(self, Xtest):
        Array = []
        for learner in self.learner:
            Array.append(learner.query(Xtest))
        return np.mean( Array, axis= 0)
