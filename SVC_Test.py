from sklearn.svm import SVR
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_excel("dummy3.xlsx")  # Einlesen der Daten


#scale = StandardScaler()
#df_sc = scale.fit_transform(df)
#df_sc = pd.DataFrame(df_sc,columns=df.columns)

y = df['EtaS']
X = df.drop('EtaS',axis=1)

Xtrain,XTest,ytrain,ytest = train_test_split(X,y,test_size=0.7,random_state=42)

SVM_regression = SVR()
SVM_regression.fit(Xtrain,ytrain)

