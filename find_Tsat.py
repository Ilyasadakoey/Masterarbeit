import numpy as np
import pandas as pd
import fluid_properties_rp as fprop


df = pd.read_excel("Molefractions.xlsx")

x1 = df["x1"]
x2 = df["x2"]
x3 = df["x3"]




fluid = "Isobutane*Propane*Propylene"



for (a,b,c) in zip(x1,x2,x3):
    evap = fprop.p_prop_sat(350000, fluid, composition=[a,b,c], option=1)
    print(a,b,c)
    print(evap)

