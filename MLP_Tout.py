import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier, MLPRegressor
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler


df = pd.read_excel("Datensatz.xlsx") #Einlesen der Daten

y = df['T_a'] # Output ''
X = df.iloc[:, [0,1,2,3,4]]

scaler = MinMaxScaler()
X = scaler.fit_transform(X, y)




Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, random_state=10, test_size=0.2)


NN = MLPRegressor(activation="relu", hidden_layer_sizes=(100,100,100),learning_rate='adaptive',solver='adam',alpha=0.0001)

model = NN.fit(Xtrain, Ytrain)

NN_pred = NN.predict(Xtest)

print(mean_squared_error(Ytest, NN_pred))
print(mean_absolute_error(Ytest, NN_pred))
print(r2_score(Ytest,NN_pred))

with open ('vorhergesagt.txt','a') as f:
     for v in NN_pred:

        f.write(str(v)+ '\n')

with open('wahr.txt', 'a') as f:
    for v in Ytest:
        f.write(str(v) + '\n')

plt.scatter(Ytest,NN_pred,s=5)
plt.plot([min(Ytest), max(Ytest)], [min(NN_pred), max(NN_pred)], linestyle='--', color='red', label='1:1-Linie')
plt.xlabel('Wahrer Wert',fontsize = 12)
plt.ylabel('Vorhergesagter Wert', fontsize = 12)
plt.title('Austrittstemperatur',fontsize = 12)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

plt.show()

dump(model,'Tout_MLP.pkl')



