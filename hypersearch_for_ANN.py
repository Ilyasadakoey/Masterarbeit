import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier, MLPRegressor
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from scipy.stats import loguniform
from sklearn.preprocessing import MinMaxScaler

# find the best hyperparameters for the ANN model


# load the simulated data

Data = pd.read_excel("Datensatz.xlsx")

# define inputs and outputs; depends on which model should be used


X = Data.iloc[:, [0, 1, 2, 3, 4]]

Y = Data['T_a']

# scale

scaler = MinMaxScaler()

X = scaler.fit_transform(X, Y)

# split the data

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, random_state=10, test_size=0.2)

# run the hyperparameter search

mlp_gs = MLPRegressor()
parameter_space = {
    'hidden_layer_sizes': [(25), (25, 25), (25, 25, 25), (25, 25, 25, 25), (25, 25, 25, 25, 25), (50), (50, 50),
                           (50, 50, 50), (50, 50, 50, 50), (50, 50, 50, 50, 50), (75), (75, 75), (75, 75, 75),
                           (75, 75, 75, 75), (75, 75, 75, 75, 75), (100), (100, 100), (100, 100, 100),
                           (100, 100, 100, 100), (100, 100, 100, 100, 100), (125), (125, 125), (125, 125, 125),
                           (125, 125, 125, 125), (125, 125, 125, 125, 125), (150), (150, 150), (150, 150, 150),
                           (150, 150, 150, 150), (150, 150, 150, 150, 150), (175), (175, 175), (175, 175, 175),
                           (175, 175, 175, 175), (175, 175, 175, 175, 175), (200), (200, 200), (200, 200, 200),
                           (200, 200, 200, 200), (200, 200, 200, 200, 200, 200), (225), (225, 225), (225, 225, 225),
                           (225, 225, 225, 225), (225, 225, 225, 225, 225, 225), (250), (250, 250), (250, 250, 250),
                           (250, 250, 250, 250), (250, 250, 250, 250, 250, 250), (275), (275, 275), (275, 275, 275),
                           (300), (300, 300), (300, 300, 300), (300, 300, 300, 300), (300, 300, 300, 300, 300, 300),
                           (325), (325, 325), (325, 325, 325), (325, 325, 325, 325, 325),
                           (325, 325, 325, 325, 325, 325), (350), (350, 350), (350, 350, 350), (350, 350, 350, 350),
                           (350, 350, 350, 350, 350)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
}

clf = RandomizedSearchCV(mlp_gs, parameter_space, n_iter=30, cv=5, verbose=2, scoring='neg_mean_squared_error')


clf.fit(Xtrain, Ytrain)

# print the found parameters

print(f"beste Parameter: {clf.best_params_}")
