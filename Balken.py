import matplotlib.pyplot as plt
import numpy as np

# Daten für das Balkendiagramm
categories = ['x\u2081', 'x\u2082', 'ΔT', 'Π','p\u1D62']
values1 = [0.0022,0.0003,0.1869,0.543,0.3339]
values2 = [0.0013,0.0001,0.1702,0.5064,0.3263]



x = np.arange(len(categories))

width = 0.35

# Erstelle das Balkendiagramm
fig,ax = plt.subplots()

bar1 = ax.bar(x - width/2, values1, width, label='S1')
bar2 = ax.bar(x + width/2, values2, width, label='ST')

plt.rcParams["font.family"]="Arial"
# Titel und Beschriftungen
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)



ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()


# Zeige das Diagramm an

save_path = 'C:\\Users\\ilyas\\OneDrive\\Desktop\\'
plt.savefig(save_path+'Sens_Tout.png',dpi=500)
plt.show()




