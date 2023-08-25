import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier, MLPRegressor
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler


df = pd.read_excel("Datensatz.xlsx") #Einlesen der Daten


real = df['EtaS']

pred = df['etaS_']

n = np.sum(real)

mean = n/len(real)

print(mean)

rmse = np.sqrt(mean_squared_error(real,pred))

print(rmse)

