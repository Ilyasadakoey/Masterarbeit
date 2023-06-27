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

Data = pd.read_excel("dummy2.xlsx")  # Einlesen der Daten

X = Data.iloc[:, 0:-2]  # Inputs: Pe,Te, Molenbrüche und Druckverhältnis

Y = Data.iloc[:, -2:]  # Outputs: Isentroper Wirkungsgrad und Liefergrad

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, random_state=3, test_size=0.7)


scaler = StandardScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)

print(Xtrain.shape)
print(Xtest.shape)
print(Ytest.shape)
print(Ytrain.shape)

model = MLPRegressor
kernel = ["linear", "rbf", "sigmoid", "poly"]
tolerance = loguniform(1e-6, 1e-3)
C = [1, 1.5, 2, 2.5, 3]
grid = dict(kernel=kernel, tol=tolerance, C=C)

print("[INFO] grid searching over the hyperparameters...")
cvFold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
randomSearch = RandomizedSearchCV(estimator=model, n_jobs=-1,
                                  cv=cvFold, param_distributions=grid,
                                  scoring="neg_mean_squared_error", error_score='raise')

searchResults = randomSearch.fit(Xtrain, Ytrain)

print("[INFO] evaluating...")
bestModel = searchResults.best_estimator_
print("R2: {:.2f}".format(bestModel.score(Xtest, Ytest)))