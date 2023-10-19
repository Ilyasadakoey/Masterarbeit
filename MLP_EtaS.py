import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier, MLPRegressor
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler

# Creates an ANN for the isentropic efficiency

# load the simulated data
df = pd.read_excel("Datensatz.xlsx") #Einlesen der Daten

# define inputs and outputs

y = df['EtaS'] # Output ''
X = df.iloc[:, [0,1,2,9]]


# scale inputs

scaler = MinMaxScaler()
X = scaler.fit_transform(X, y)

# split inputs and outputs

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, random_state=10, test_size=0.2)

# run the regressor, hyperparameters depend on hypersearch results
NN = MLPRegressor(activation="relu", hidden_layer_sizes=(300,300,300),learning_rate='adaptive',solver='adam',alpha=0.0001)

# train the model
model = NN.fit(Xtrain, Ytrain)

# test the model

NN_pred = NN.predict(Xtest)

# calculate RSME and R²

n = np.sum(Ytrain)
mean = n/len(Ytrain)

print(np.sqrt(mean_squared_error(Ytest, NN_pred)))
print(mean_absolute_error(Ytest, NN_pred))
print(r2_score(Ytest,NN_pred))
print(mean)

# plot and save the results

plt.rcParams["font.family"]="Arial"
plt.scatter(Ytest,NN_pred,s=5)
plt.plot([min(Ytest), max(Ytest)], [min(NN_pred), max(NN_pred)], linestyle='--', color='red', label='1:1-Linie')
plt.xlabel('Wahrer Wert / -',fontsize = 11)
plt.ylabel('Vorhergesagter Wert / -', fontsize = 11)
plt.title('Isentroper Wirkungsgrad mit der Eintrittsdichte als zusätzliche Größe',fontsize = 11)
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)

save_path = 'C:\\Users\\ilyas\\OneDrive\\Desktop\\'
plt.savefig(save_path+'MLPforEtaS_rho',dpi=500)

plt.show()

# save the model to use it later

dump(model,'EtaS_MLP_rho.pkl')



