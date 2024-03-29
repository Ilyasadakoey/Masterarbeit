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

T_in = df['Tin']
p_in = df['p_e']
xa = df['xa']
#xb = df['xb']

for i,v in enumerate(T_in):

    v_spez = rp.tp(T_in[i],p_in[i]*1000,"Isobutane*Propane",composition=[xa[i],1-xa[i]],option=1,units=_units,props=_props)[3]

    with open('rho_in_bin.txt','a') as f:
             f.write(str(1/v_spez)+'\n')

