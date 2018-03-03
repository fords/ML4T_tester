import numpy as np
import DTLearner as dt
import matplotlib.pyplot as plt
import time
import matplotlib.patches as mpatches
import BagLearner as bl
import RTLearner as rtl
if __name__ == '__main__':
    data = np.genfromtxt('Data/Istanbul.csv', delimiter = ',');
    data = data[1:,1:];# eliminate 1st row and 1st column containing col-labels and dates
    #np.random.shuffle(data)# shuffle in-place
    split = int(0.6*data.shape[0])#60-40 break into train-test sets
    trainX = data[:split,:-1]
    trainY = data[:split,-1]#last column is labels
    testX = data[split:,:-1]
    testY = data[split:,-1]#last column is labels

    # create a learner and train it
    start = time.time()
    #DT_learner =rtl.RTLearner(leaf_size = 1, verbose = True) # create an DTLearner
    #DT_learner.addEvidence(trainX, trainY)# train it
    learner = bl.BagLearner(learner = dt.DTLearner, kwargs =  {"leaf_size":1} ,bags = 20, boost = False, verbose = False)

    learner.addEvidence(trainX, trainY)# train it
#
    # in sample testing
    Y = learner.query(trainX)# get the predictions
    rmse = ( np.sqrt( ( (Y - trainY)**2 )/trainY.shape[0] ) ).sum()
    corr = np.corrcoef(Y, trainY)
    print("In sample results")
    print("RMSE: ", rmse)
    print("corr: ", corr[0,1])

    # out of sample testing
    Y = learner.query(testX)# get the predictions
    rmse = ( np.sqrt( ( (Y - testY)**2 )/testY.shape[0] ) ).sum()
    corr = np.corrcoef(Y, testY)
    print
    print("Out of sample results")
    print("RMSE: ", rmse)
    print("corr: ", corr[0,1])
    stop = time.time()
    print 'time = {} s'.format(stop - start)

    # Plot RMSE for training dataset vs. leaf_size
    maxLeafSize = 50
    errTrain = np.zeros(maxLeafSize);
    errTest = np.zeros(maxLeafSize);
    for size in range(1, maxLeafSize+1):
        learner = bl.BagLearner(learner =dt.DTLearner,  kwargs =  {"leaf_size":1}, bags = 20, boost = False, verbose = False)# create learner
        for i in range(10):
            #np.random.shuffle(data) # shuffle in-place
            learner.addEvidence(trainX, trainY) # train on shuffled trainX, trainY
            # training sample testing
            Y = learner.query(trainX)# get the predictions
            errTrain[size-1] += np.sqrt( ( (Y - trainY)**2 ).sum() / trainY.shape[0] )
            # test sample testing
            Y = learner.query(testX)# get the predictions
            errTest[size-1] += np.sqrt( ( (Y - testY)**2 ).sum() / testY.shape[0] )


    plt.plot(range(1, maxLeafSize+1), errTrain/100, label = 'training')
    plt.plot(range(1, maxLeafSize+1), errTest/100, label = 'test')
    plt.xlabel('leaf size'); plt.ylabel('RMS error')
    plt.title(' Bag Learner')
    red_patch = mpatches.Patch(color='blue', label='Test')
    orange_patch = mpatches.Patch(color='red', label='Training')
    plt.legend(handles=[orange_patch,red_patch])
    plt.show()
    lg = plt.legend(loc = 'best')
    lg.draw_frame(False)
