import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel("Pred_EtaS.xlsx")

x = df ["Real"]

y = df ["Predcit"]

plt.scatter(x,y,s=5)

plt.plot([min(x), max(x)], [min(y), max(y)], linestyle='--', color='gray', label='1:1-Linie')


plt.xlim(0)
plt.ylim(0)

plt.show()