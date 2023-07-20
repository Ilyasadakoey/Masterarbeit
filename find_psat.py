import numpy as np
import pandas as pd
import fluid_properties_rp as fprop

df = pd.read_excel("Molefractions.xlsx")

x1 = df["x1"]
x2 = df["x2"]
x3 = df["x3"]

Tevap = 273.15
T_oh = 5
Tin = Tevap + T_oh
fluid = "Isobutane*Propane*Propylene"

for (a, b, c) in zip(x1, x2, x3):
    evap = fprop.T_prop_sat(Tevap, fluid, composition=[a, b, c], option=1)
    psat = evap[1,1]
    print(a,b,c)
    print(psat)
    ptxt = str(psat)
    with open('data.txt', 'a') as f:
        f.write('\n')
        f.write(ptxt)
