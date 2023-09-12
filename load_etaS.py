import joblib as jb
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel("binarymixture2.xlsx")

x = df.iloc[:, [0,1,2]]
y = df['etaS']

scaler = MinMaxScaler()
X = scaler.fit_transform(x)

loaded_model = jb.load('EtaA_MLP_sens.pkl')



predictions = loaded_model.predict(X)


n = np.sum(y)
mean = n/len(y)

print(np.sqrt(mean_squared_error(y, predictions)))
print(mean_absolute_error(y, predictions))
print(r2_score(y,predictions))
print(mean)



