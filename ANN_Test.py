import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier, MLPRegressor
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import dump, load

Data = pd.read_excel("dummy2.xlsx") #Einlesen der Daten

X = Data.iloc[:, 0:-2]  # Inputs: Pe,Te, Molenbrüche und Druckverhältnis

Y = Data.iloc[:, -2:]  # Outputs: Isentroper Wirkungsgrad und Liefergrad

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, random_state=0, test_size=0.7)

NN = MLPRegressor(max_iter=300, activation="relu", hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100))

NN.fit(Xtrain, Ytrain)

NN_pred = NN.predict(Xtest)

print(mean_squared_error(Ytest, NN_pred))
print(mean_absolute_error(Ytest, NN_pred))


dump(NN,'test123.joblib')

loaded_model =load('test123.joblib')

