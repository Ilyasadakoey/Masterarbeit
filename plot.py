import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel("Isobutane_Propan_EtaS.xlsx")



x = df['p2/p1']
y = df['EtaS']
Z = df ['xa']

# Streudiagramm mit Farbskala erstellen
scatter = plt.scatter(x, y, c=Z, cmap='cividis',s=10)  # 'viridis' ist eine Farbskala, Sie können eine andere auswählen
plt.colorbar(scatter, label='Werte der Farbskala')

# Achsenbeschriftungen
plt.xlabel('Variable X')
plt.ylabel('Variable Y')

# Titel hinzufügen
plt.title('Streudiagramm mit Farbskala')

# Diagramm anzeigen
plt.show()