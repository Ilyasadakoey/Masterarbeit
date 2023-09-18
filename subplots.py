import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

df = pd.read_excel("pred_binary_eta.xlsx")



x = df ['true_eta_rho']

y1 = df ['pred_eta_sens']

y2 = df['pred_eta_rho']

plt.subplot(1,2,1)
plt.scatter(x,y1,label='sens')


plt.subplot(1,2,2)
plt.scatter(x,y2,label='rho')

plt.show()




plt.tight_layout()



plt.show()




