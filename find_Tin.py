import numpy as np
import matplotlib.pyplot as plt
from fl_props_compressor import z_uv, z_ps, z_Tp, z_Tx, z_mm
import fluid_properties_rp as rp
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
import pandas as pd
import os


RP = REFPROPFunctionLibrary(os.environ['RPPREFIX'])
_props = "REFPROP"
_units = RP.GETENUMdll(0, "MASS BASE SI").iEnum



df = pd.read_excel("binarymixture2.xlsx")

p_in = df['p_e']
xa = df['xa']
dT = df['dT']

for i,v in enumerate(p_in):
    T_sat = rp.p_prop_sat(p_in[i]*1000,fluid = 'Isobutane*Propane',composition=(xa[i],1-xa[i]),option=1,units=_units,props=_props)[0]
    T_in = T_sat[0]+dT[i]
    print(T_in)

    with open('T_in_bin.txt', 'a') as f:
        f.write(str(T_in) + '\n')
