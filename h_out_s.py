import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fl_props_compressor import z_uv, z_ps, z_Tp, z_Tx, z_mm
import fluid_properties_rp as rp
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
import os

RP = REFPROPFunctionLibrary(os.environ['RPPREFIX'])
_props = "REFPROP"
_units = RP.GETENUMdll(0, "MASS BASE SI").iEnum

df = pd.read_excel('Datensatz.xlsx')

T_in = df['T_e']
p_in = df['p_e']
p_ve = df['p_ve']
xa = df['xa']
xb = df['xb']
sin = df['s_in']

for i,v in enumerate(sin):
    h_outs = rp.sp(sin[i],p_in[i]*p_ve[i]*1000,"Isobutane*Propane*Propylene",composition=[xa[i],xb[i],1-xa[i]-xb[i]],option=1,units=_units,props=_props)[2]

    with open('h_outs.txt', 'a') as f:
        f.write(str(h_outs/1000) + '\n')