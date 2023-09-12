import joblib as jb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel("binarymixture2.xlsx")

x = df.iloc[:, [0,1,2]]
y = df['Taus']

scaler = MinMaxScaler()
X = scaler.fit_transform(x)

loaded_model = jb.load('Tout_MLP.pkl')



predictions = loaded_model.predict(X)


n = np.sum(y)
mean = n/len(y)

print(np.sqrt(mean_squared_error(y, predictions)))
print(mean_absolute_error(y, predictions))
print(r2_score(y,predictions))
print(mean)

print(predictions)

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

plt.scatter(y,predictions,s=5)
plt.plot([min(y), max(y)], [min(predictions), max(predictions)], linestyle='--', color='red', label='1:1-Linie')

save_path = 'C:\\Users\\ilyas\\OneDrive\\Desktop\\'
plt.savefig(save_path+'Pred_Tout_binary',dpi=500)


plt.show()