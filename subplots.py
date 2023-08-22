import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

df = pd.read_excel("Datensatz.xlsx")



x = df['T_e']
y = df['p_e']
Z1 = df ['xa']
Z2 = df['dT']

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8,6))

im1 = ax1.scatter(x,y,c=Z1,cmap='viridis')
ax1.set_xlabel('Eintrittstemperatur / K',fontsize = 10)
ax1.set_ylabel('Eintrittsdruck / kPa', fontsize = 10)


cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('Molenbruch Isobutan / -',fontsize = 10)
#cbar1.ax.yaxis.set_label_position('right')
ax1.tick_params(axis='both', which='major', labelsize=12)
cbar1.ax.tick_params(labelsize=12)

im2 = ax2.scatter(x,y,c=Z2,cmap = 'viridis')
ax2.set_xlabel('Eintrittstemperatur / K',fontsize = 10)
ax2.set_ylabel('Eintrittsdruck / kPa', fontsize = 10)
ax2.tick_params(axis='both', which='major', labelsize=12)



cbar2 = plt.colorbar(im1, ax=ax1)
cbar2.set_label('Überhitzung / K',fontsize = 10)
#cbar2.ax.yaxis.set_label_position('right')
cbar2.ax.tick_params(labelsize=12)


plt.tight_layout()

plt.show()




