import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

df = pd.read_excel("Datensatz.xlsx")



x = df['T_e']
y = df['p_e']
Z1 = df ['xa']
Z2 = df['dT']

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8.27,11.69))

im1 = ax1.scatter(x,y,c=Z1,cmap='viridis')
ax1.set_xlabel('Eintrittstemperatur / K',fontsize = 15)
ax1.set_ylabel('Eintrittsdruck / kPa', fontsize = 15)



divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right",size="5%",pad=0.05)
cbar1 = plt.colorbar(im1, cax=cax1)
cbar1.set_label('Molenbruch Isobutan / -',fontsize = 15)
ax1.tick_params(axis='both', which='major', labelsize=15)
cbar1.ax.tick_params(labelsize=15)

ax1.set_xlim(0,400)
ax1.set_ylim(0)
ax1.grid(True)

im2 = ax2.scatter(x,y,c=Z2,cmap = 'viridis')
ax2.set_xlabel('Eintrittstemperatur / K',fontsize = 15)
ax2.set_ylabel('Eintrittsdruck / kPa', fontsize = 15)
ax2.tick_params(axis='both', which='major', labelsize=15)



divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right",size="5%",pad=0.05)
cbar2 = plt.colorbar(im2, cax=cax2)
cbar2.set_label('Überhitzung / K',fontsize = 15)
cbar2.ax.tick_params(labelsize=15)
ax2.set_xlim(0,400)
ax2.set_ylim(0)
ax2.grid(True)


plt.tight_layout()



plt.show()




