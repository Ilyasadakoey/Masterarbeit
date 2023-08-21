# -*- coding: utf-8 -*-
"""
___________________________________________________________________________________________________

SensAnalysis_example

    "xample of a sensitivity analysi based on the polygeneration process concept v 4.5.
    The process concept itself was removed and only sensitivity analysis function is still included."

    v1.0        01.02.2023      built example python file


    --------------------------------------------------------------------------------------------

    created on Wed Jan 24 12:27:07 2018

    author:     Dominik Freund
___________________________________________________________________________________________________

"""
import multiprocessing

"==========================================   IMPORT   ==========================================="

from SALib.sample import saltelli
from SALib.analyze import sobol
from scipy.optimize import root, fsolve

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import sys
import os

<<<<<<< HEAD

=======
>>>>>>> origin/main
"==========================================   CLASS   ============================================"


class SensAnalysis(object):

    # =============================================================================================
    #       Initialization
    # =============================================================================================
    def __init__(self):

        self.plots = True
        self.file_name = os.path.basename(sys.argv[0]).strip('.py')

        # =========================================================================================
        #       Excel sheet path
        # =========================================================================================

<<<<<<< HEAD


=======
>>>>>>> origin/main
        # =========================================================================================
        #       Sensitivity analysis
        # =========================================================================================
        self.N_sample = 2 ** 1  # number of samples (input) created for S.A.
        self.cut_off = 0.005  # threshold for important S.A. results
<<<<<<< HEAD


=======
        self.input_data = {'num_vars': 5, 'names': ['dT', 'p_ve', 'p_e', 'a', 'b'],
                           'bounds': [[2, 25], [2, 8], [200, 600], [0.5, 0.7], [0.05, 0.29]]}
        self.param_values = saltelli.sample(problem=self.input_data, N=self.N_sample, calc_second_order=False)
        self.results = []
>>>>>>> origin/main



    "====================================   FUNCTIONS   =========================================="

<<<<<<< HEAD

    # =================================================================================================

    def sensitivity_analysis(self):

       # self.date = str((datetime.datetime.now()).strftime("%Y%m%d-%H%M%S"))
        #self.file_out = 'Results/' + self.date + '/'
        #os.makedirs(self.file_out)

        # inputs for SA
        self.problem = {'num_vars':3,'names':['dT','p_ve','p_e'], 'bounds':[[2,25],[2,8],[200,600]],}

        self.param_values = saltelli.sample(problem=self.problem, N=self.N_sample, calc_second_order=False)
        self.results = []

        #print("\n______________________________________________\nrun sensitivity analyis....\n")
        for i in range(len(self.param_values[:, 0])):
=======
    # =================================================================================================

    def sensitivity_analysis(self,input_data):

        # self.date = str((datetime.datetime.now()).strftime("%Y%m%d-%H%M%S"))
        # self.file_out = 'Results/' + self.date + '/'
        # os.makedirs(self.file_out)



        # inputs for SA





        # print("\n______________________________________________\nrun sensitivity analyis....\n")
        for i in range(len(self.param_values[:, 0])):

>>>>>>> origin/main
            self.param_values_SA = self.param_values[i, :]

            # print("param_values_SA: ",self.param_values_SA)

            # run the model
            res_temp = self.model(self.param_values_SA)  # example only, no function

            self.results.append(res_temp)

        self.results = np.array(self.results)
        self.results_SA = {}
        self.result_labels = ['label'] * int(len((self.results)[0, :]))

        for r in range(len(self.results[0, :])):
<<<<<<< HEAD
            #print("\n >>> ", self.result_labels[r], "\n")

            self.results_SA[self.result_labels[r]] = sobol.analyze(self.problem, self.results[:, r],
=======
            # print("\n >>> ", self.result_labels[r], "\n")

            self.results_SA[self.result_labels[r]] = sobol.analyze(self.input_data, self.results[:, r],
>>>>>>> origin/main
                                                                   calc_second_order=False)

            self.si = self.results_SA[self.result_labels[r]]

            self.i_highSens1 = np.where(self.si["S1"] > self.cut_off)[0]
            self.i_highSensT = np.where(self.si["ST"] > self.cut_off)[0]

            if len(self.i_highSens1) > 0:

                for j, i in enumerate(self.i_highSens1):
<<<<<<< HEAD



=======
>>>>>>> origin/main
                    # print results in console
                    self.oii = "{0:2d}, {1:20s} S1: {2:= 9.4f}, ST: {3:= 9.4f}" \
                        .format(i, "dummy", self.si["S1"][i], self.si["ST"][i])
                    sensis = str(self.oii)
<<<<<<< HEAD
                    with open ('senis.txt','a') as f:
                        f.write("\n")
                        f.write(sensis)
                        f.write("\n")
                    #print(self.oii)

            #else:
                #print("\n no sensitivity found")

=======
                    with open('senis.txt', 'a') as f:
                        f.write("\n")
                        f.write(sensis)
                        f.write("\n")
                    # print(self.oii)

            # else:
            # print("\n no sensitivity found")

    def run_sensitivity_analysis(self, input_chunk):
        results = []
        for input_data in input_chunk:
            result = self.sensitivity_analysis(input_data)
            results.append(result)

        return results
>>>>>>> origin/main
    # =================================================================================================

    def model(self, args):
        from compressor_roskosch_orig_rp import getETA
        import fluid_properties_rp as rp
        from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
        import os
        RP = REFPROPFunctionLibrary(os.environ['RPPREFIX'])
        _props = "REFPROP"
        _units = RP.GETENUMdll(0, "MASS BASE SI").iEnum

<<<<<<< HEAD
        dT,p_ve,p_e = args
        y = getETA(dT, p_ve, p_e, fluid_in='Isobutane * Propane * Propylene', comp=[1,0,0],
                   pV=[34e-3, 34e-3, 3.5, .04, .06071, 48.916, 50., 50. / 2., 2.], pZ=np.zeros(7, float),
                   z_it=np.zeros([360, 16]), IS=360, pZyk=np.zeros(2, float), IS0=360)
        T_e = dT + rp.p_prop_sat(p=p_e*1000, fluid='Isobutane * Propane * Propylene', composition=[1,0,0],
                                 option=1, units=_units, props=_props)[0, 0]

        out = str([p_ve,p_e,dT,T_e,y[0],y[1],y[2]])

        with open('results.txt','a') as f:
            f.write("\n")
            f.write(out)
            f.write("\n")
=======

        dT, p_ve, p_e, a, b = args
        y = getETA(dT, p_ve, p_e, fluid_in='Isobutane * Propane*Propylene', comp=[a, b, 1 - a - b],
                   pV=[34e-3, 34e-3, 3.5, .04, .06071, 48.916, 50., 50. / 2., 2.], pZ=np.zeros(7, float),
                   z_it=np.zeros([360, 16]), IS=360, pZyk=np.zeros(2, float), IS0=360)

        T_e = dT + rp.p_prop_sat(p=p_e * 1000, fluid='Isobutane * Propane*Propylene', composition=[a, b, 1 - a - b],
                                 option=1, units=_units, props=_props)[
            0, 0]

        # out = str([dT, p_e, T_e, p_ve, y[0], y[1], y[2]])

        dTtxt = str(dT)
        p_veTxt = str(p_ve)
        p_e = str(p_e)
        Tetxt = str(T_e)
        atxt = str(a)
        btxt = str(b)
        EtaStxt = str(y[0])
        LambdaLtxt = str(y[1])
        Taustxt = str(y[2])

        with open('Results.txt', 'a') as f:
            f.write("\n")
            f.write(dTtxt), f.write("  "), f.write(p_veTxt), f.write("  "), f.write(p_e), f.write("  "), f.write(
                atxt), f.write("  "), f.write(btxt), f.write("  "), f.write(EtaStxt), f.write("  "), f.write(
                LambdaLtxt), f.write("  "), f.write(
                Taustxt), f.write("  "), f.write(Tetxt)
>>>>>>> origin/main

        return y


<<<<<<< HEAD

class Parallel:
    def __int__(self,num_processes):

        self.num_processes = 4

    def run_parallel_sensitivity_analysis(self):
        pool = multiprocessing.Pool(processes=4)

        instances = [SensAnalysis() for _ in
                        range(4)]  # Erstelle Instanzen der SensAnalysis-Klasse

        # Nutze die `map`-Funktion, um die Sensitivitätsanalyse auf die Instanzen aufzuteilen
        pool.map(SensAnalysis.sensitivity_analysis, instances)

        pool.close()
        pool.join()


=======
>>>>>>> origin/main
"==========================================   RUN   =============================================="

if __name__ == "__main__":
    import time

    # import datetime
    s = time.time()

    # =============================================================================
    # Sensitivity analysis
    # =============================================================================

<<<<<<< HEAD

dddd



=======
>>>>>>> origin/main
    SA = SensAnalysis()
    SA.N_sample = 2 ** 1  # number of samples (2^n)
    SA.cut_off = -1e5  # 0.01   # threshold for important S.A. results

<<<<<<< HEAD
    parallel_analysis = Parallel()
    parallel_analysis.run_parallel_sensitivity_analysis()

    # run sensitivity analysis




    SA.sensitivity_analysis()

    problem = SA.problem
=======
    # run sensitivity analysis

    num_processes = 2
    input_data_list = SA.param_values



    chunk_size = len(input_data_list) // num_processes
    input_chunks = [input_data_list[i:i + chunk_size] for i in range(0, len(input_data_list), chunk_size)]
    pool = multiprocessing.Pool(processes=num_processes)
    results = pool.map(SA.run_sensitivity_analysis, input_chunks)
    pool.close()
    pool.join()
    print(input_chunks)


    SA.sensitivity_analysis(input_chunks)

    problem = SA.input_data
>>>>>>> origin/main
    results = SA.results
    sensIndex = SA.si
    paramVal = SA.param_values

    results_SA = SA.results_SA

<<<<<<< HEAD
    #import pickle

    #pickle.dump(problem, open(SA.file_out + SA.date + "_sa_problem.p", "wb"))
    #pickle.dump(results, open(SA.file_out + SA.date + "_sa_results.p", "wb"))
   #pickle.dump(paramVal, open(SA.file_out + SA.date + "_sa_paramVal.p", "wb"))
    #pickle.dump(SA.result_labels, open(SA.file_out + SA.date + "_sa_result_labels.p", "wb"))
    #pickle.dump(results_SA,open(SA.file_out + SA.date+ "_sa_sensis.p","wb"))

    e = time.time()
    print("\nRuntime = {} s ({} h)".format(np.round(e - s, 1), np.round((e - s) / 3600, 2)))
=======
    # import pickle

    # pickle.dump(problem, open(SA.file_out + SA.date + "_sa_problem.p", "wb"))
    # pickle.dump(results, open(SA.file_out + SA.date + "_sa_results.p", "wb"))
    # pickle.dump(paramVal, open(SA.file_out + SA.date + "_sa_paramVal.p", "wb"))
    # pickle.dump(SA.result_labels, open(SA.file_out + SA.date + "_sa_result_labels.p", "wb"))
    # pickle.dump(results_SA,open(SA.file_out + SA.date+ "_sa_sensis.p","wb"))

    e = time.time()
    # print("\nRuntime = {} s ({} h)".format(np.round(e - s, 1), np.round((e - s) / 3600, 2)))
>>>>>>> origin/main