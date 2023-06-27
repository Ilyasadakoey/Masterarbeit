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

Y = Data.iloc[:, -1:]  # Outputs: Isentroper Wirkungsgrad und Liefergrad

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, random_state=3, test_size=0.7)


mlp_gs = MLPClassifier(max_iter=1)
parameter_space = {
    'hidden_layer_sizes': [(10,),(20,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5,error_score='raise')


Y = np.ravel(Y)
clf.fit(X, Y)