import numpy as np
import sys
from sklearn.metrics import zero_one_loss

training_data = np.loadtxt("classification_train.txt")

X_train = training_data[:,:-1]
y_train = [int(z) for z in training_data[:,-1]]

# baseline model, ignore values of X_train, just use the majority class
y_train_majority_class = np.argmax(np.bincount(y_train))

if len(sys.argv) > 1:
    test_data = np.loadtxt(sys.argv[1])
    X_test = test_data[:,:-1]
    y_test = [int(z) for z in test_data[:,-1]]

    # ignore values of X_test in test data, just use training set majority class
    y_test_pred = [y_train_majority_class] * len(y_test)
    loss = zero_one_loss(y_test,y_test_pred)
    print("Classification accuracy for baseline is {0}".format(1-loss))
