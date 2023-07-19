import numpy as np
import pandas as pd
import fluid_properties_rp as fprop


x1 = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
x2 = [0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26]

x3 = 1 - np.array(x2)-np.array(x1)

Tin = 278.15
Tsat = Tin - 5
fluid = "Isobutane*Propane*Propylene"


for (a,b,c) in zip(x1,x2,x3):
    evap = fprop.T_prop_sat(Tsat, fluid, composition=[a,b,c], option=1)
    print(a,b,c)
    print(evap[1,1])



