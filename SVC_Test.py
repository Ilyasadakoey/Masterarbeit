from sklearn.svm import SVR
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

Data = pd.read_excel("dummy2.xlsx")  # Einlesen der Daten

X = Data.iloc[:, 0:-2]  # Inputs: Pe,Te, Molenbrüche und Druckverhältnis

Y = Data.iloc[:, -2:]  # Outputs: Isentroper Wirkungsgrad und Liefergrad

#Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, random_state=42, test_size=0.7)


sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = np.ravel(Y)

regressor = SVR(kernel='rbf')


regressor.fit(X,Y)

