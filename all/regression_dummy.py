import numpy as np
import sys
from sklearn.metrics import mean_squared_error

training_data = np.loadtxt("regression_train.txt")

X_train = training_data[:,:-1]
y_train = training_data[:,-1]

# baseline model, ignore values of X, just use the mean
y_train_mean = np.mean(y_train)

if len(sys.argv) > 1:
    test_data = np.loadtxt(sys.argv[1])
    X_test = test_data[:,:-1]
    y_test = test_data[:,-1]

    # ignore values of X in test data, just use training set mean
    y_test_pred = [y_train_mean] * len(y_test)
    mse = mean_squared_error(y_test,y_test_pred)
    print("MSE for baseline is {0}".format(mse))
