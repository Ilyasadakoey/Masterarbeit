import joblib as jb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel("binarymixture2.xlsx")

x = df.iloc[:, [0,1,2,9]]
y = df['Taus']
m = df['xa']
scaler = MinMaxScaler()
X = scaler.fit_transform(x)

loaded_model = jb.load('Tout_MLP_rho.pkl')



predictions = loaded_model.predict(X)


n = np.sum(y)
mean = n/len(y)

print(np.sqrt(mean_squared_error(y, predictions)))
print(mean_absolute_error(y, predictions))
print(r2_score(y,predictions))
print(mean)

print(predictions)

Intervall_start = 0.50
Intervall_ende = 0.70

# Punkte im Intervall filtern
in_interval = (m >= Intervall_start) & (m <= Intervall_ende)

# Punkte außerhalb des Intervalls
out_of_interval = ~in_interval

# Punkte innerhalb des Intervalls in rot plotten
plt.scatter(y[in_interval], predictions[in_interval], c='blue',s=5, label='Innerhalb Intervall')

# Punkte außerhalb des Intervalls in schwarz plotten
plt.scatter(y[out_of_interval], predictions[out_of_interval], c='black',s=5, label='Außerhalb Intervall')

plt.rcParams["font.family"]="Arial"
plt.xlabel('Wahrer Wert / K',fontsize = 11)
plt.ylabel('Vorhergesagter Wert / K', fontsize = 11)
plt.title('Austrittstemperatur für ein binäres Gemisch',fontsize = 11)
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)

plt.plot([min(y), max(y)], [min(predictions), max(predictions)], linestyle='--', color='red', label='1:1-Linie')
save_path = 'C:\\Users\\ilyas\\OneDrive\\Desktop\\'
plt.savefig(save_path+'Pred_Tout_binary_alle',dpi=500)


plt.show()