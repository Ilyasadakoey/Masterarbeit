import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ps1 = 185.77  # saturation pressure at 5°C of Isobutane

ps2 = 548.69  # saturation pressure at 5°C of Propane

ps3 = 673.28  # saturation pressure at 5°C of Propylene


def p(x1):
    return x1 * ps2 + (1 - x1) * ps1


def y(x1):
    return (x1 * ps2) / p(x1)


x1 = np.linspace(0, 1, 100)

plt.rc('axes',labelsize=11)
plt.rc('axes',titlesize=11)
plt.rc('legend',fontsize=11)
plt.rc('xtick',labelsize=11)
plt.rc('ytick',labelsize=11)
plt.rc('font',size=11)
plt.plot(x1, p(x1),label='Siedelinie',linewidth=3)
plt.plot(y(x1),p(x1),'r',label='Taulinie',linewidth=3,)
plt.ylim(0,600)
plt.xlabel('Molenbruch Propan')
plt.ylabel('Druck [kPa]')
plt.text(0.4,500,'Flüssigphase',style='normal')
plt.text(0.4,150,'Gasphase',style='normal')
#plt.title('p-xy Diagramm von Isobutan-Propan bei 5°C')
plt.legend()
plt.show()

