import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel("Datensatz.xlsx")

x = df['T_e']
y = df['p_e']
Z = df['xa']

# Streudiagramm mit Farbskala erstellen

scatter = plt.scatter(x, y, c=Z, cmap='cividis', s=10)
colorbar = plt.colorbar(scatter, label='Molenbruch Isobutan / -')

colorbar.set_label('Molenbruch Isobutan / -', fontsize=12)

plt.grid(True)

colorbar.ax.yaxis.set_label_coords(4.5, 0.5)

plt.xlim(0)
plt.ylim(0)

# Achsenbeschriftungen
plt.xlabel('Eintrittstemperatur / K', fontsize=12)
plt.ylabel('Eintrittsdruck / kPa', fontsize=12)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

colorbar.ax.tick_params(labelsize=12)

plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)

# Diagramm anzeigen



save_path = 'C:\\Users\\ilyas\\OneDrive\\Desktop\\'
plt.savefig(save_path+'test',dpi=500)
