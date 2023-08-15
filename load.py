import joblib as jb
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor

df = pd.read_excel("EtaS_Out.xlsx")

x = df.drop('EtaS',axis = 1)

loaded_model = jb.load('test123.pkl')

predictions = loaded_model.predict(x)


print(predictions)
