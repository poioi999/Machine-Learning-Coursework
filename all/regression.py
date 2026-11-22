import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import RandomizedSearchCV
import pymc as pm
import arviz as az
from itertools import product
import torch
from torch import nn
from torch.nn import functional as F
from typing import Callable
from torch import optim


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
print("Train MSE Linear Regression: ", round(MSE_train, 2))

MSE_test = (mean_squared_error(y_test, model.predict(X_test)))
print("Test MSE Linear Regression: ", round(MSE_test, 2))
'''


'''
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y_test, model.predict(X_test))
ax.plot([0,5],[0,5])
ax.set_xlabel('Real value')
ax.set_ylabel('Prediction')
plt.show()
'''

'''
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model.fit(X_train_poly, y_train)

y_test_pred = model.predict(X_test_poly)

MSE_train = (mean_squared_error(y_train, model.predict(X_train_poly)))
print("Train MSE Polynomial Regression: ", round(MSE_train, 2))

MSE_test = (mean_squared_error(y_test, y_test_pred))
print("Test MSE Polynomial Regression: ", round(MSE_test, 2))
'''

'''
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

# neural network regression using PyTorch


X_train_nn = torch.tensor(X_train, dtype=torch.float32)
y_train_nn = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test_nn = torch.tensor(X_test, dtype=torch.float32)
y_test_nn = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

class MLP(nn.Module):
    def __init__(self,  input_size:int, hidden_layer_size: int, output_size:int, activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.tanh):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_layer_size)
        self.l2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.l3 = nn.Linear(hidden_layer_size, output_size)
        self.activation_fn = activation_fn

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.l1(inputs)
        x = self.activation_fn(x)
        x = self.l2(x)
        x = self.activation_fn(x)
        x = self.l3(x)
        return x

feature_count = 3
hidden_layer = 8
output_size = 1

nn_model = MLP(feature_count, hidden_layer, output_size)

logits = nn_model.forward(X_train_nn)
print(logits.shape)

loss_function = nn.MSELoss()
#loss = loss_function(logits, y_train_nn)

#loss.backward()

optimizer = optim.Adam(nn_model.parameters(), lr = 0.001, weight_decay = 0.01)

loss_vals_before = []

for epoch in range(1, 1000):
    logits = nn_model.forward(X_train_nn)
    loss = loss_function(logits, y_train_nn)

    print("epoch: {}, loss: {:5.5f}".format(
        epoch,
        loss.item()
    ))
    loss_vals_before.append(loss.item())

    loss.backward()
    optimizer.step()

    optimizer.zero_grad()

logits = nn_model.forward(X_test_nn)    
test_mse = loss_function(logits, y_test_nn).item()

print(f'Final Testing MSE (PyTorch): {test_mse:.4f}')

# Bayesian regression using PyMC

'''
print(X_train.shape)


num_samples = 1000
bayesian_model = pm.Model()

with bayesian_model:
    # define priors 

    w0 = pm.Normal('w0', mu=0, sigma=10)
    w1 = pm.Normal('w1', mu=0, sigma=10)
    w2 = pm.Normal('w2', mu=0, sigma=10)
    w3 = pm.Normal('w3', mu=0, sigma=10)
    sigma = pm.Uniform('sigma', lower=0, upper=20)

    y_est = w0 + w1 * X_train[:, 0] + w2 * X_train[:, 1] + w3 * X_train[:, 2]

    likelihood = pm.Normal('y', mu=y_est, sigma=sigma, observed=y_train)

    sampler = pm.NUTS()

    idata = pm.sample(num_samples, step=sampler, progressbar=True)

#pc = az.plot_trace_dist(idata, compact=False)
#pc.show()

w0_est = idata.posterior['w0'].mean().item()
w1_est = idata.posterior['w1'].mean().item()
w2_est = idata.posterior['w2'].mean().item()
w3_est = idata.posterior['w3'].mean().item()

def predict(X):
    return w0_est + w1_est * X[:, 0] + w2_est * X[:, 1] + w3_est * X[:, 2]

y_train_pred = predict(X_train)
y_test_pred = predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print("Bayesian Linear Regression Training MSE:", train_mse)
print("Bayesian Linear Regression Testing MSE:", test_mse)


# Bayesian polynomial regression

num_features = X_train_poly.shape[1]
samples = 1000

with pm.Model() as bayesian_poly_model:

    w0 = pm.Normal('w0', mu=0, sigma=20)
    w = pm.Normal('w', mu=0, sigma=20, shape=num_features)
    sigma = pm.Uniform('sigma', lower=0, upper=20)

    y_est = w0 + pm.math.dot(X_train_poly, w)

    likelihood = pm.Normal('y', mu=y_est, sigma=sigma, observed=y_train)

    idata_poly = pm.sample(samples, tune=1000, chains=4, step=pm.NUTS(), progressbar=True)

w0_est = idata_poly.posterior['w0'].mean().item()
w_est = idata_poly.posterior['w'].mean(dim=['chain', 'draw']).values

def predict_poly(X):
    return w0_est + np.dot(X, w_est)

y_train_pred = predict_poly(X_train_poly)
y_test_pred = predict_poly(X_test_poly)

print("Bayesian Polynomial Regression Training MSE:", mean_squared_error(y_train, y_train_pred))
print("Bayesian Polynomial Regression Testing MSE:", mean_squared_error(y_test, y_test_pred))

#pc = az.plot_trace_dist(idata_poly, compact=False)
#pc.show()
'''