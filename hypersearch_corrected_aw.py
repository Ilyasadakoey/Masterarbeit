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

Data = pd.read_excel("Datensatz.xlsx")  # Einlesen der Daten

X = Data.iloc[:, [0,1,2,9]]  # Inputs: Pe,Te, Molenbrüche und Druckverhältnis

Y = Data['T_a']  # Outputs: Isentroper Wirkungsgrad und Liefergrad


scaler = MinMaxScaler()

X = scaler.fit_transform(X,Y)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, random_state=10, test_size=0.2)

mlp_gs = MLPRegressor(max_iter=1000)
parameter_space = {
    'hidden_layer_sizes': [(75), (75,75), (75,75,75),(100), (100,100), (100,100,100),(250),(250,250),(250,250,250),(300),(300,300),(300,300,300)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
}
from sklearn.model_selection import GridSearchCV

clf = RandomizedSearchCV(mlp_gs, parameter_space, n_iter = 30, cv=5, verbose = 2,scoring='neg_mean_squared_error')

#Ytrain2 = np.ravel(Ytrain)
clf.fit(Xtrain, Ytrain)

print(f"beste Parameter: {clf.best_params_}")
