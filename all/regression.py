import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV


training_data = np.loadtxt("regression_train.txt")

X = training_data[:,:-1]
y = training_data[:,-1]

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
print('Training set size:', len(X_train))
print('Test set size:', len(X_test))

'''
model = LinearRegression()

model.fit(X_train, y_train)

MSE_train = (mean_squared_error(y_train, model.predict(X_train)))
print("Train MSE: ", round(MSE_train, 2))

MSE_test = (mean_squared_error(y_test, model.predict(X_test)))
print("Test MSE: ", round(MSE_test, 2))

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y_test, model.predict(X_test))
ax.plot([0,5],[0,5])
ax.set_xlabel('Real value')
ax.set_ylabel('Prediction')
plt.show()


poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model.fit(X_train_poly, y_train)

y_test_pred = model.predict(X_test_poly)

MSE_train = (mean_squared_error(y_train, model.predict(X_train_poly)))
print("Train MSE: ", round(MSE_train, 2))

MSE_test = (mean_squared_error(y_test, y_test_pred))
print("Test MSE: ", round(MSE_test, 2))

min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())


fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y_test, y_test_pred)
ax.plot([min_val, max_val], [min_val, max_val])
ax.set_xlabel('Real value')
ax.set_ylabel('Prediction')
plt.show()


#print(model.score(X_test_poly, y_test))
'''

nn_clf = MLPRegressor(
    hidden_layer_sizes=(32,32,), # started with 1 layer 128
    activation='tanh', # started with relu
    solver='adam', # started with sgd
    learning_rate_init=np.float64(0.0007390722033525775), # started with 0.001
    alpha=np.float64(0.00187381742286030), # started with default
    max_iter=500, 
    n_iter_no_change=25,
    random_state=42
)


nn_clf.fit(X_train, y_train)


fig, ax = plt.subplots(figsize=(6,4))
ax.plot(nn_clf.loss_curve_)
ax.set_xlabel('Number of iterations')
ax.set_ylabel('Loss')

plt.show()


y_train_pred = nn_clf.predict(X_train)
y_test_pred = nn_clf.predict(X_test)

print('Training MSE: ', round(mean_squared_error(y_train, y_train_pred), 4))
print('Testing MSE: ', round(mean_squared_error(y_test, y_test_pred), 4))



'''
alphas = np.logspace(-3, 0, 100)
learning_rates = np.logspace(-4, -2, 100)

hidden_layers = [
    (32,),
    (64,),
    (128,),
    (32, 32),
    (64, 32),
    (64, 64),
    (128, 64),
    (64, 32, 16)
]

iterations = [300, 400, 500, 525, 600, 750, 1000, 1500, 2000]

activations = ['identity', 'logistic', 'tanh', 'relu']

solvers = ['lbfgs', 'sgd', 'adam']

parameters = {
    'learning_rate_init': learning_rates,
    'alpha': alphas,
    'hidden_layer_sizes': hidden_layers,
    'activation': activations,
    'solver': solvers,
    'max_iter': iterations
}

clf = RandomizedSearchCV(
    nn_clf,
    parameters,
    scoring='neg_mean_squared_error',
    n_iter=40,
    cv=5,
    verbose=2,
    random_state=42
)

clf.fit(X_train, y_train)

best_nn = clf.best_estimator_

print('Best parameters:', clf.best_params_)
print('Best CV MSE:', -clf.best_score_)

y_train_pred = best_nn.predict(X_train)
y_test_pred = best_nn.predict(X_test)

#print('Training MSE:', mean_squared_error(y_train, y_train_pred))
#print('Testing MSE:', mean_squared_error(y_test, y_test_pred))
'''