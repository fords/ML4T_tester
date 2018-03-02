import numpy as np
import LinRegLearner, DTLearner, RTLearner


class BagLearner(object):

    def __init__(self, learner,kwargs, bags=20, boost=False, verbose=False ):
        """Initalize a Bag Learner
        Parameters:
        learner: A LinRegLearner, DTLearner, or RTLearner
        bags: The number of learners to be trained using Bootstrap Aggregation
        boost: If true, boosting will be implemented
        verbose: If True, information about the learner will be printed out
        kwargs: Keyword arguments to be passed on to the learner's constructor

        Returns: An instance of Bag Learner
        """
        self.verbose = verbose
        learners = []
        for i in range(bags):
            learners.append(learner(kwargs))
        self.learners = learners
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        if verbose:
            self.get_learner_info()


    def addEvidence(self, dataX, dataY):
        """Add training data to learner
        Parameters:
        dataX: A numpy ndarray of X values to add
        dataY: A numpy 1D array of Y values to add
        Returns: Updated individual learners in BagLearner
        """
        # Sample the data with replacement
        num_samples = dataX.shape[0]
        for learner in self.learners:
            idx = np.random.choice(num_samples, num_samples)
            bagX = dataX[idx]
            bagY = dataY[idx]
            learner.addEvidence(bagX, bagY)
        if self.verbose:
            self.get_learner_info()


    def query(self, points):
        """Estimates a set of test points given the model we built

        Parameters:
        points: A numpy ndarray of test queries
        Returns:
        preds: A numpy 1D array of the estimated values
        """
        preds = np.array([learner.query(points) for learner in self.learners])
        return np.mean(preds, axis=0)


    def get_learner_info(self):
        """Print out data for this BagLearner"""
        learner_name = str(type(self.learners[0]))[8:-2]
        print ("This BagLearner is made up of {} {}:".
            format(self.bags, learner_name))

        print ("kwargs =", self.kwargs)
        print ("boost =", self.boost)

        # Print out information for each learner within BagLearner
        for i in range(1, self.bags + 1):
            print (learner_name, "#{}:".format(i));
            self.learners[i-1].get_learner_info()


if __name__=="__main__":
    print ("This is a Bag Learner\n")

    # Some data to test the BagLearner
    x0 = np.array([0.885, 0.725, 0.560, 0.735, 0.610, 0.260, 0.500, 0.320])
    x1 = np.array([0.330, 0.390, 0.500, 0.570, 0.630, 0.630, 0.680, 0.780])
    x2 = np.array([9.100, 10.900, 9.400, 9.800, 8.400, 11.800, 10.500, 10.000])
    x = np.array([x0, x1, x2]).T

    y = np.array([4.000, 5.000, 6.000, 5.000, 3.000, 8.000, 7.000, 6.000])

    # Create a BagLearner from given training x and y
    bag_learner = BagLearner(DTLearner.DTLearner, verbose=True)

    print ("\nAdd data")
    bag_learner.addEvidence(x, y)

    # Query with dummy data
    print ("Query with dummy data:\n", np.array([[1, 2, 3], [0.2, 12, 12]]))
    print ("Query results:", bag_learner.query(np.array([[1, 2, 3], [0.2, 12, 12]])))
