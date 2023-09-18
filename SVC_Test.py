from sklearn.svm import SVR
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_excel("LambdaL_Out.xlsx")  # Einlesen der Daten


#scale = StandardScaler()
#df_sc = scale.fit_transform(df)
#df_sc = pd.DataFrame(df_sc,columns=df.columns)

y = df['LambdaL'] # Output ''
X = df.drop('LambdaL',axis=1)

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.7,random_state=42)

SVM_regression = SVR()
SVM_regression.fit(Xtrain,ytrain)

y_hat = SVM_regression.predict(Xtest)

print(mean_squared_error(ytest, y_hat))
print(mean_absolute_error(ytest, y_hat))
