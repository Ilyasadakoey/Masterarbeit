import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel("Pred_EtaS.xlsx")

x = df ["Real"]

y = df ["Predcit"]

plt.rcParams["font.family"]="Arial"

plt.grid(True)

plt.scatter(x,y,s=5)



plt.plot([min(x), max(x)], [min(y), max(y)], linestyle='--', color='red', label='1:1-Linie')

plt.text(0.05,0.9,'0,6631 ± 0,0396',transform=plt.gca().transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

plt.xlabel('Isentroper Wirkungsgrad (Simulation) / -', fontsize=11)
plt.ylabel('Isentroper Wirkungsgrad (Händisch) / -', fontsize=11)

plt.xticks(fontsize=11)
plt.yticks(fontsize=11)



plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)

save_path = 'C:\\Users\\ilyas\\OneDrive\\Desktop\\'
plt.savefig(save_path+'EtaS_error.png',dpi=500)





plt.show()