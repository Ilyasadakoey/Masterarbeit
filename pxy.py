import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ps1 = 185.77  # saturation pressure at 5°C of Isobutane

ps2 = 548.69  # saturation pressure at 5°C of Propane

ps3 = 673.28  # saturation pressure at 5°C of Propylene


def p(x1):
    return x1 * ps1 + (1 - x1) * ps2


def y(x1):
    return (x1 * ps1) / p(x1)


x1 = np.linspace(0, 1, 100)

plt.plot(x1, p(x1))
plt.plot(y(x1),p(x1))
plt.show()
