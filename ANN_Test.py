import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier, MLPRegressor
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import dump, load

df = pd.read_excel("EtaS_Out.xlsx") #Einlesen der Daten

y = df['EtaS'] # Output ''
X = df.drop('EtaS',axis=1)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, random_state=42, test_size=0.7)


NN = MLPRegressor(max_iter=1000, activation="tanh", hidden_layer_sizes=(100,),learning_rate='adaptive',solver='sgd',alpha=0.0001)

model = NN.fit(Xtrain, Ytrain)

NN_pred = NN.predict(Xtest)

print(mean_squared_error(Ytest, NN_pred))
print(mean_absolute_error(Ytest, NN_pred))

dump(model,'test123.pkl')



