import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel("Datensatz.xlsx")

x = df['p_ve']
y = df['T_a']
Z = df['rho_in']

# Streudiagramm mit Farbskala erstellen

plt.rcParams["font.family"]="Arial"

scatter = plt.scatter(x, y, c=Z, cmap='cividis', s=10)
colorbar = plt.colorbar(scatter, label='Eintrittsdichte / kg/m³')

colorbar.set_label('Eintrittsdichte / kg/m³', fontsize=12)

plt.grid(True)

colorbar.ax.yaxis.set_label_coords(4.5, 0.5)

plt.xlim(0)
plt.ylim(0)

# Achsenbeschriftungen
plt.xlabel('Druckverhältnis / -', fontsize=12)
plt.ylabel('Austrittstemperatur / K', fontsize=12)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

colorbar.ax.tick_params(labelsize=12)

plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)

# Diagramm anzeigen




save_path = 'C:\\Users\\ilyas\\OneDrive\\Desktop\\'
plt.savefig(save_path+'Tout_rhoin.png',dpi=500)


plt.show()