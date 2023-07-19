import numpy as np
import pandas as pd
import fluid_properties_rp as fprop

df = pd.read_excel("Molefractions.xlsx")

x1 = df["x1"]
x2 = df["x2"]
x3 = df["x3"]

Tin = 278.15
Tsat = Tin - 5
fluid = "Isobutane*Propane*Propylene"

for (a, b, c) in zip(x1, x2, x3):
    evap = fprop.T_prop_sat(Tsat, fluid, composition=[a, b, c], option=1)
    psat = evap[1, 1]
    ptxt = str(psat)
    with open('data.txt', 'a') as f:
        f.write('\n')
        f.write(ptxt)
