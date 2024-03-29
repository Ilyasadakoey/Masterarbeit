import multiprocessing

"==========================================   IMPORT   ==========================================="

from SALib.sample import saltelli
from SALib.analyze import sobol
from scipy.optimize import root, fsolve
import multiprocessing
import concurrent.futures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import sys
import os
from compressor_roskosch_orig_rp import getETA
import fluid_properties_rp as rp
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
import os

RP = REFPROPFunctionLibrary(os.environ['RPPREFIX'])
_props = "REFPROP"
_units = RP.GETENUMdll(0, "MASS BASE SI").iEnum

# runs the Roskosch Model with the saltelli-sequenz

N_sample = 2 ** 11
cut_off = 0.005
results = []
input_data = {'num_vars': 5, 'names': ['dT', 'p_ve', 'p_e', 'xa', 'xb'],
              'bounds': [[2, 25], [2, 8], [200, 600], [0.5, 0.7], [0.05, 0.29]]}
param_values = saltelli.sample(problem=input_data, N=N_sample, calc_second_order=False)


def sensitivity_analysis(param_values):
    results = []
    cut_off = 0.005

    for i in range(len(param_values[:, 0])):
        param_values_SA = param_values[i, :]

        res_temp = model(param_values_SA)

        results.append(res_temp)

    results = np.array(results)
    results_SA = {}
    result_labels = ['label'] * int(len((results)[0, :]))
    for r in range(len(results[0, :])):
        # print("\n >>> ", self.result_labels[r], "\n")

        results_SA[result_labels[r]] = sobol.analyze(input_data, results[:, r],
                                                     calc_second_order=False)

        si = results_SA[result_labels[r]]

        i_highSens1 = np.where(si["S1"] > cut_off)[0]
        i_highSensT = np.where(si["ST"] > cut_off)[0]

        if len(i_highSens1) > 0:

            for j, i in enumerate(i_highSens1):
                # print results in console
                oii = "{0:2d}, {1:20s} S1: {2:= 9.4f}, ST: {3:= 9.4f}" \
                    .format(i, "dummy", si["S1"][i], si["ST"][i])
                sensis = str(oii)
                with open('senis.txt', 'a') as f:
                    f.write("\n")
                    f.write(sensis)
                    f.write("\n")
    return results


def model(args):
    dT, p_ve, p_e, xa, xb = args
    y = getETA(dT, p_ve, p_e, fluid_in='Isobutane * Propane*Propylene', comp=[xa, xb, 1 - xa - xb],
               pV=[34e-3, 34e-3, 3.5, .04, .06071, 48.916, 50., 50. / 2., 2.], pZ=np.zeros(7, float),
               z_it=np.zeros([360, 16]), IS=360, pZyk=np.zeros(2, float), IS0=360)

    T_e = dT + rp.p_prop_sat(p=p_e * 1000, fluid='Isobutane * Propane*Propylene', composition=[xa, xb, 1 - xa - xb],
                             option=1, units=_units, props=_props)[
        0, 0]

    dTtxt = str(dT)
    p_veTxt = str(p_ve)
    p_e = str(p_e)
    Tetxt = str(T_e)
    atxt = str(xa)
    btxt = str(xb)
    EtaStxt = str(y[0])
    LambdaLtxt = str(y[1])
    Taustxt = str(y[2])

    with open('Results.txt', 'a') as f:
        f.write("\n")
        f.write(dTtxt), f.write("  "), f.write(p_veTxt), f.write("  "), f.write(p_e), f.write("  "), f.write(atxt),
        f.write("  "), f.write(btxt), f.write(EtaStxt), f.write("  "), f.write(
            LambdaLtxt), f.write("  "), f.write(
            Taustxt), f.write("  "), f.write(Tetxt)

    return y


# parallelize the simulation #

def process_task(param_values_chunk):
    results = []

    for i in range(len(param_values_chunk)):
        param_values = param_values_chunk[i, :]
        res_temp = model(param_values)
        results.append(res_temp)


if __name__ == "__main__":
    import time

    s = time.time()

    num_processes = multiprocessing.cpu_count()

    chunk_size = len(param_values) // num_processes

    param_chunks = [param_values[i:i + chunk_size] for i in range(0, len(param_values), chunk_size)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_task, chunk) for chunk in param_chunks]

    e = time.time()
    with open('Time.txt', 'a') as f:
        f.write("\nRuntime = {} s ({} h)".format(np.round(e - s, 1), np.round((e - s) / 3600, 2)))
