import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier, MLPRegressor
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler

# create an ANN for the outlet temperature


# load data from simulation
df = pd.read_excel("Datensatz.xlsx")  # Einlesen der Daten

# define inputs and outputs


y = df['T_a']  # Output ''
X = df.iloc[:, [0, 1, 2, 9]]

# scale

scaler = MinMaxScaler()
X = scaler.fit_transform(X, y)

# split data


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, random_state=10, test_size=0.2)

# run the regressor

NN = MLPRegressor(activation="tanh", hidden_layer_sizes=(75), learning_rate='adaptive', solver='adam', alpha=0.0001,
                  max_iter=1000)

# train the model

model = NN.fit(Xtrain, Ytrain)

# test the model

NN_pred = NN.predict(Xtest)

# calculate RMSE and R²


n = np.sum(Ytrain)
mean = n / len(Ytrain)

np.set_printoptions(threshold=np.inf)

rs = np.reshape(NN_pred, (2750, 1))

with open('T_predict.txt', 'a') as f:
    f.write(str(rs))

print(np.sqrt(mean_squared_error(Ytest, NN_pred)))
print(mean_absolute_error(Ytest, NN_pred))
print(r2_score(Ytest, NN_pred))
print(mean)

# plot the results

plt.rcParams["font.family"] = "Arial"

plt.scatter(Ytest, NN_pred, s=5)
plt.plot([min(Ytest), max(Ytest)], [min(NN_pred), max(NN_pred)], linestyle='--', color='red', label='1:1-Linie')
plt.xlabel('Wahrer Wert / K', fontsize=11)
plt.ylabel('Vorhergesagter Wert / K', fontsize=11)
plt.title('Austrittstemperatur mit der Eintrittsdichte als zusätzliche Größe', fontsize=11)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)

save_path = 'C:\\Users\\ilyas\\OneDrive\\Desktop\\'
plt.savefig(save_path + 'MLPforTout_rhoin', dpi=500)

plt.show()

# save the model

dump(model, 'Tout_MLP_rho.pkl')
